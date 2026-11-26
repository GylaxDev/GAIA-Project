"""Mesh extraction, procedural setbacks, complex roofs, and rooftop assets."""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient

from .config import MESH_CACHE_VERSION, ROAD_WIDTHS
from .data_fetcher import (Building, Road, GreenSpace, build_trees_mesh, parse_tile_elements,
                           _clean_coords, _jitter, theme_code, _color_for_theme)
from .projection import LocalProjection

try:
    import mapbox_earcut as earcut
except ImportError:
    earcut = None


@dataclass
class MeshData:
    vertices: np.ndarray
    indices: np.ndarray
    edges: np.ndarray
    radius: float
    max_height: float


@dataclass
class ChunkResult:
    key: tuple
    b_mesh: MeshData | None
    g_mesh: MeshData | None
    w_mesh: MeshData | None
    trees: np.ndarray | None
    coarse: bool = False


def edge_color(wall_col):
    return np.clip(wall_col * 2.4 + np.array([0.04, 0.10, 0.18], np.float32), 0, 1).astype(np.float32)


def ring_vertices(coords):
    if len(coords) > 1 and coords[0, 0] == coords[-1, 0] and coords[0, 1] == coords[-1, 1]: coords = coords[:-1]
    if len(coords) < 2: return np.empty((0, 2), np.float64)
    keep = np.linalg.norm(coords - np.roll(coords, -1, 0), axis=1) > 1e-6
    return coords[keep]


class MeshExtruder:
    SIMPLIFY_TOL = 0.25
    SMOOTH_COS = np.cos(np.radians(35))
    SHARP_COS = np.cos(np.radians(25))

    def build(self, buildings, roads, greens, waters, trees):
        b_mesh = self._build_buildings(buildings) if buildings else None
        g_mesh = self._build_ground(roads, greens) if (roads or greens) else None
        w_mesh = self._build_water(waters) if waters else None
        t_mesh = build_trees_mesh(trees) if trees else None
        return b_mesh, g_mesh, w_mesh, t_mesh

    def _build_buildings(self, buildings):
        V, I, lines = [], [], []
        nv = 0

        def push_batch(v, i, l):
            nonlocal nv
            if len(v) == 0: return
            V.append(v)
            I.append(i + nv)
            lines.append(l)
            nv += len(v)

        for b in buildings:
            if b.parts:
                for p in b.parts:
                    self._process_volume(p, p.footprint, p.height, p.min_height, push_batch)
            else:
                tiers = []
                if b.height > 60 and b.roof_shape == "flat":
                    num_tiers = min(3, int(b.height // 40))
                    tier_h = b.height / (num_tiers + 1)
                    current_poly = b.footprint
                    
                    for i in range(num_tiers):
                        z0 = b.min_height + i * tier_h
                        z1 = z0 + tier_h
                        tiers.append((current_poly, z0, z1))
                        inset = min(2.0 + 1.5 * i, 6.0)
                        shrunk = current_poly.buffer(-inset)
                        if shrunk.is_empty or shrunk.area < 20.0:
                            break
                        current_poly = shrunk
                        
                    tiers.append((current_poly, b.min_height + num_tiers * tier_h, b.height))
                    
                    for poly, z0, z1 in tiers:
                        self._process_volume(b, poly, z1, z0, push_batch)
                        
                    self._build_roof_assets(b, current_poly, b.height, b.roof_shape, b.roof_height, push_batch)
                else:
                    self._process_volume(b, b.footprint, b.height, b.min_height, push_batch)
                    self._build_roof_assets(b, b.footprint, b.height, b.roof_shape, b.roof_height, push_batch)

        if not V: return None
        vertices = np.ascontiguousarray(np.vstack(V), dtype=np.float32)
        indices = np.ascontiguousarray(np.concatenate(I), dtype=np.uint32)
        edges = np.ascontiguousarray(np.concatenate(lines), dtype=np.float32) if lines else np.empty((0, 6), np.float32)
        
        radius = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
        return MeshData(vertices, indices, edges, radius, float(vertices[:, 2].max()))

    def _process_volume(self, bld, poly, z1, z0, push_batch):
        if poly.is_empty: return
        poly = orient(poly, 1.0)
        
        rings = []
        for ring in [poly.exterior] + list(poly.interiors):
            r = ring_vertices(np.asarray(ring.coords))
            if len(r) >= 3: rings.append(r)
        if not rings: return

        style_arr = np.array([_jitter(bld.id), theme_code(bld.kind, bld.id), z1], np.float32)
        
        wall_v, wall_i, l = self._walls_batch(
            rings, np.full(len(rings), z0), np.full(len(rings), z1),
            np.tile(bld.color, (len(rings), 1)),
            np.tile(edge_color(bld.color), (len(rings), 1)),
            np.tile(style_arr, (len(rings), 1))
        )
        push_batch(wall_v, wall_i, l)

        if z1 == bld.height and bld.roof_shape == "flat":
            pts, tris = self._triangulate(poly)
            if len(tris) > 0:
                a, bb, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
                cross = ((bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
                tris = np.where((cross < 0)[:, None], tris[:, [0, 2, 1]], tris)
                
                roof = np.empty((len(pts), 15), np.float32)
                roof[:, 0:2] = pts; roof[:, 2] = z1
                roof[:, 3:6] = (0.0, 0.0, 1.0)
                roof[:, 6:9] = bld.roof_color
                roof[:, 9] = 1.0
                roof[:, 10:13] = style_arr
                roof[:, 11] = 4.0
                push_batch(roof, tris.ravel(), np.empty((0, 6), np.float32))

    def _build_roof_assets(self, bld, poly, z1, shape, roof_h, push_batch):
        if poly.is_empty: return
        style_arr = np.array([_jitter(bld.id), 4.0, z1], np.float32)
        
        if shape in ("pyramidal", "dome") and roof_h > 0:
            c = poly.centroid
            pts = np.asarray(poly.exterior.coords)[:-1]
            apex = np.array([c.x, c.y, z1 + roof_h], np.float32)
            
            n = len(pts)
            v = np.zeros((n + 1, 15), np.float32)
            v[:n, 0:2] = pts; v[:n, 2] = z1
            v[n] = [apex[0], apex[1], apex[2], 0, 0, 1, *bld.roof_color, 1.0, *style_arr, 0, 0]
            
            idx = []
            for i in range(n):
                p0, p1 = v[i, :3], v[(i+1)%n, :3]
                nx, ny, nz = np.cross(p1 - p0, apex - p0)
                norm = np.linalg.norm([nx, ny, nz])
                if norm > 0: v[i, 3:6] = v[(i+1)%n, 3:6] = [nx/norm, ny/norm, nz/norm]
                idx += [i, (i+1)%n, n]
            
            push_batch(v, np.array(idx, np.uint32), np.empty((0, 6), np.float32))
            
        elif shape == "pitched" and roof_h > 0:
            minx, miny, maxx, maxy = poly.bounds
            w, h = maxx - minx, maxy - miny
            if w > h:
                r0, r1 = np.array([minx, (miny+maxy)/2, z1 + roof_h]), np.array([maxx, (miny+maxy)/2, z1 + roof_h])
            else:
                r0, r1 = np.array([(minx+maxx)/2, miny, z1 + roof_h]), np.array([(minx+maxx)/2, maxy, z1 + roof_h])
            
            v = np.zeros((6, 15), np.float32)
            v[0, :3] = r0; v[1, :3] = r1; v[2, :3] = [minx, miny, z1]
            v[3, :3] = [maxx, maxy, z1]; v[4, :3] = [minx, maxy, z1]; v[5, :3] = [maxx, miny, z1]
            v[:, 6:9] = bld.roof_color; v[:, 10:13] = style_arr
            push_batch(v, np.array([0,1,2, 0,2,3, 0,3,4, 0,4,5], np.uint32), np.empty((0, 6), np.float32))

        if shape == "flat" and z1 > 30:
            cx, cy = poly.centroid.x, poly.centroid.y
            size = 4.0 if z1 < 100 else 8.0
            box = Polygon([(cx-size, cy-size), (cx+size, cy-size), (cx+size, cy+size), (cx-size, cy+size)])
            box = box.intersection(poly)
            if not box.is_empty and box.area > 10:
                self._process_volume(bld, box, z1 + 4.0, z1, push_batch)
                
        if shape == "flat" and z1 > 80:
            cx, cy = poly.centroid.x, poly.centroid.y
            spire_h = 15.0 + _jitter(bld.id) * 30.0
            v = np.zeros((5, 15), np.float32)
            v[0, :3] = [cx, cy, z1 + spire_h]
            v[1, :3] = [cx-0.8, cy-0.8, z1]
            v[2, :3] = [cx+0.8, cy-0.8, z1]
            v[3, :3] = [cx+0.8, cy+0.8, z1]
            v[4, :3] = [cx-0.8, cy+0.8, z1]
            v[:, 3:6] = (0, 0, 1)
            v[:, 6:9] = (0.8, 0.8, 0.9)
            v[:, 10:13] = style_arr
            v[:, 11] = 5.0
            push_batch(v, np.array([0,1,2, 0,2,3, 0,3,4, 0,4,1], np.uint32), np.empty((0, 6), np.float32))

    def _walls_batch(self, rings, z0, z1, col, edge_col, style, want_lines=True):
        valid_rings, valid_z0, valid_z1, valid_col, valid_ecol, valid_style = [], [], [], [], [], []
        for r, z0v, z1v, cv, ev, sv in zip(rings, z0, z1, col, edge_col, style):
            if np.isfinite(r).all() and np.max(np.hypot(r[:,0], r[:,1])) < 5000.0:
                valid_rings.append(r); valid_z0.append(z0v); valid_z1.append(z1v)
                valid_col.append(cv); valid_ecol.append(ev); valid_style.append(sv)
        if not valid_rings:
            return np.empty((0, 15), np.float32), np.empty(0, np.uint32), np.empty((0, 6), np.float32)
            
        rings = valid_rings; z0 = np.array(valid_z0); z1 = np.array(valid_z1)
        col = np.array(valid_col); edge_col = np.array(valid_ecol); style = np.array(valid_style)

        counts = np.fromiter((len(r) for r in rings), np.int64, len(rings))
        starts = np.concatenate(([0], np.cumsum(counts)))
        v = np.concatenate(rings); n = len(v)
        rep = np.repeat(np.arange(len(counts)), counts)
        z0v, z1v = z0[rep], z1[rep]
        colv, ecolv = col[rep], edge_col[rep]; stylev = style[rep]
        idx = np.arange(n); nxt = idx + 1
        nxt[starts[1:] - 1] = starts[:-1]
        prv = idx - 1; prv[starts[:-1]] = starts[1:] - 1
        p0, p1 = v, v[nxt]
        d = p1 - p0; d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        en = np.stack([d[:, 1], -d[:, 0]], axis=1); prev_n = en[prv]
        def blend(a, other):
            s = a + other; s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-9)
            return np.where(((a * other).sum(1) > self.SMOOTH_COS)[:, None], s, a)
        n_start, n_end = blend(en, prev_n), blend(en, en[nxt])
        dp, dn = d[prv], d[nxt]
        ao_s = np.where(dp[:, 0] * d[:, 1] - dp[:, 1] * d[:, 0] < -1e-3, 0.78, 1.0)
        ao_e = np.where(d[:, 0] * dn[:, 1] - d[:, 1] * dn[:, 0] < -1e-3, 0.78, 1.0)
        
        L = np.linalg.norm(p1 - p0, axis=1)
        
        w = np.empty((n, 4, 15), np.float32)
        w[:, 0, 0:2], w[:, 3, 0:2] = p0, p0
        w[:, 1, 0:2], w[:, 2, 0:2] = p1, p1
        w[:, 0, 2], w[:, 1, 2] = z0v, z0v
        w[:, 2, 2], w[:, 3, 2] = z1v, z1v
        w[:, 0, 3:5], w[:, 3, 3:5] = n_start, n_start
        w[:, 1, 3:5], w[:, 2, 3:5] = n_end, n_end
        w[:, :, 5] = 0.0; w[:, :, 6:9] = colv[:, None, :]
        w[:, 0, 9], w[:, 3, 9] = ao_s, ao_s
        w[:, 1, 9], w[:, 2, 9] = ao_e, ao_e
        w[:, :, 10:13] = stylev[:, None, :]
        
        w[:, 0, 13] = 0; w[:, 1, 13] = L; w[:, 2, 13] = L; w[:, 3, 13] = 0
        w[:, 0, 14] = L; w[:, 1, 14] = L; w[:, 2, 14] = L; w[:, 3, 14] = L
        
        quad = np.arange(n)[:, None] * 4
        wi = (quad + np.array([0, 1, 2, 0, 2, 3])).ravel()
        if not want_lines:
            return w.reshape(-1, 15), wi, np.empty((0, 6), np.float32)
            
        roof = np.empty((n, 2, 6), np.float32)
        roof[:, 0, 0:2], roof[:, 1, 0:2] = p0, p1
        roof[:, :, 2] = z1v[:, None]
        roof[:, :, 3:6] = ecolv[:, None, :]
        sharp = (en * prev_n).sum(1) < self.SHARP_COS
        vert = np.empty((int(sharp.sum()), 2, 6), np.float32)
        if sharp.any():
            vert[:, :, 0:2] = v[sharp][:, None, :]
            vert[:, 0, 2], vert[:, 1, 2] = z0v[sharp], z1v[sharp]
            vert[:, :, 3:6] = ecolv[sharp][:, None, :] * 0.55
        lines = np.concatenate([roof, vert]).reshape(-1, 6)
        return w.reshape(-1, 15), wi, lines

    def _triangulate(self, poly):
        if earcut is not None:
            try:
                rings = [np.asarray(poly.exterior.coords)[:-1]]
                rings += [np.asarray(h.coords)[:-1] for h in poly.interiors]
                pts = np.ascontiguousarray(np.vstack(rings), dtype=np.float64)
                ends = np.cumsum([len(x) for x in rings]).astype(np.uint32)
                tris = np.asarray(earcut.triangulate_float64(pts, ends), dtype=np.int64).reshape(-1, 3)
                tris = tris[(tris >= 0).all(axis=1) & (tris < len(pts)).all(axis=1)]
                if len(tris): return pts, tris
            except Exception: pass
        return self._triangulate_fallback(poly)

    @staticmethod
    def _triangulate_fallback(poly):
        tri_polys = []
        cdt = getattr(shapely, "constrained_delaunay_triangles", None)
        if cdt is not None:
            try: tri_polys = [t for t in cdt(poly).geoms if t.geom_type == "Polygon"]
            except Exception: tri_polys = []
        if not tri_polys:
            from shapely.ops import triangulate
            tri_polys = [t for t in triangulate(poly) if poly.covers(t.representative_point())]
        if not tri_polys: return np.empty((0, 2), np.float64), np.empty((0, 3), np.int64)
        pts = np.array([c for t in tri_polys for c in list(t.exterior.coords)[:3]], dtype=np.float64)
        return pts.reshape(-1, 2), np.arange(len(pts)).reshape(-1, 3)

    _JOINT_ANG = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)

    def _build_ground(self, roads, greens):
        verts_list, idx_list = [], []
        offset = 0
        def push(v, idx):
            nonlocal offset
            verts_list.append(np.ascontiguousarray(v, np.float32))
            idx_list.append(np.asarray(idx, np.uint32) + np.uint32(offset))
            offset += len(v)

        if greens:
            g_polys = []
            for g in greens:
                try:
                    p = g.geometry.buffer(0)
                    if not p.is_empty:
                        c = np.array([0.25, 0.45, 0.20]) if g.kind in ("grass", "meadow", "park", "garden") else np.array([0.15, 0.35, 0.10])
                        g_polys.append((p, c))
                except Exception: pass
            g_mesh = self._triangulate_flat(g_polys, z=0.04)
            if g_mesh: push(g_mesh.vertices, g_mesh.indices)

        for r in sorted(roads, key=lambda r: r.highway == "railway"):
            if r.highway not in ROAD_WIDTHS: continue
            w = ROAD_WIDTHS.get(r.highway, 6.0); h = w * 0.5
            is_rail = r.highway == "railway"
            z_offset = 0.09 if is_rail else 0.06
            color = (0.15, 0.12, 0.10) if is_rail else (0.06, 0.06, 0.07)

            raw = np.array(r.geometry.coords)
            if len(raw) < 2: continue
            valid = np.isfinite(raw).all(axis=1) & (np.hypot(raw[:, 0], raw[:, 1]) < 5000.0)
            segments, current = [], []
            for i in range(len(raw)):
                if valid[i] and not (current and np.linalg.norm(raw[i] - current[-1]) > 1000.0):
                    current.append(raw[i]); continue
                if len(current) >= 2: segments.append(np.array(current))
                current = [raw[i]] if valid[i] else []
            if len(current) >= 2: segments.append(np.array(current))

            for coords in segments:
                keep = np.linalg.norm(np.diff(coords, axis=0), axis=1) > 0.1
                coords = coords[np.concatenate(([True], keep))]
                if len(coords) < 2: continue
                segs = coords[1:] - coords[:-1]
                lengths = np.maximum(np.linalg.norm(segs, axis=1), 1e-4)
                d = segs / lengths[:, None]
                nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
                cum = np.concatenate(([0.0], np.cumsum(lengths)))
                m = len(segs)

                p0, p1 = coords[:-1], coords[1:]
                quad = np.empty((m, 4, 2), np.float64)
                quad[:, 0] = p0 + nrm * h; quad[:, 1] = p1 + nrm * h
                quad[:, 2] = p1 - nrm * h; quad[:, 3] = p0 - nrm * h
                v = np.zeros((m, 4, 15), np.float32)
                v[:, :, 0:2] = quad; v[:, :, 2] = z_offset
                v[:, :, 5] = 1.0; v[:, :, 6:9] = color; v[:, :, 9] = 1.0
                v[:, :, 13] = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
                v[:, 0, 14] = v[:, 3, 14] = cum[:-1]
                v[:, 1, 14] = v[:, 2, 14] = cum[1:]
                base = (np.arange(m, dtype=np.uint32) * 4)[:, None]
                push(v.reshape(-1, 15), (base + np.array([0, 1, 2, 0, 2, 3], np.uint32)).ravel())

                if m >= 2:
                    bend = np.where((d[:-1] * d[1:]).sum(axis=1) < 0.995)[0] + 1
                    if len(bend):
                        k = len(bend); c = coords[bend]
                        ring = np.stack([np.cos(self._JOINT_ANG), np.sin(self._JOINT_ANG)], 1) * (h * 1.03)
                        jv = np.zeros((k, 9, 15), np.float32)
                        jv[:, 0, 0:2] = c
                        jv[:, 1:, 0:2] = c[:, None, :] + ring[None]
                        jv[:, :, 2] = z_offset; jv[:, :, 5] = 1.0
                        jv[:, :, 6:9] = color; jv[:, :, 9] = 1.0
                        jv[:, :, 13] = 0.25
                        fan = np.array([[0, 1 + i, 1 + (i + 1) % 8] for i in range(8)], np.uint32)
                        jb = (np.arange(k, dtype=np.uint32) * 9)[:, None, None]
                        push(jv.reshape(-1, 15), (fan[None] + jb).ravel())

        if not verts_list: return None
        V = np.vstack(verts_list)
        I = np.ascontiguousarray(np.concatenate(idx_list), dtype=np.uint32)
        return MeshData(V, I, np.empty((0, 6), np.float32), float(np.max(np.hypot(V[:, 0], V[:, 1]))), 0.1)

    def _build_water(self, waters):
        if not waters: return None
        DEEP = np.array([0.059, 0.169, 0.282], np.float32)
        polys = [(p, DEEP) for p in waters if p is not None and not p.is_empty]
        return self._triangulate_flat(polys, z=0.0) if polys else None

    def _triangulate_flat(self, polys, z):
        from .data_fetcher import _parts
        verts_list, idx_list = [], []
        offset = 0
        for poly, color in polys:
            if poly.is_empty: continue
            try: poly = poly.buffer(0)
            except Exception: continue
            for p in _parts(poly):
                if p.geom_type != "Polygon" or p.is_empty or p.area < 1.0: continue
                rings = []
                for ring in [p.exterior] + list(p.interiors):
                    coords = _clean_coords(np.asarray(ring.coords)[:-1])
                    if len(coords) >= 3: rings.append(coords)
                if not rings: continue
                pts = np.ascontiguousarray(np.vstack(rings), dtype=np.float64)
                if not np.isfinite(pts).all() or not (np.hypot(pts[:, 0], pts[:, 1]) < 10000.0).all(): continue
                ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
                tris = None
                if earcut is not None:
                    try:
                        tris = np.asarray(earcut.triangulate_float64(pts, ends), dtype=np.int64).reshape(-1, 3)
                        valid = (tris >= 0).all(axis=1) & (tris < len(pts)).all(axis=1)
                        tris = tris[valid]
                        if len(tris) == 0: tris = None
                    except Exception: tris = None
                if tris is None:
                    from shapely.ops import triangulate
                    tri_polys = [t for t in triangulate(p) if p.covers(t.representative_point())]
                    if not tri_polys: continue
                    pts = np.array([c for t in tri_polys for c in list(t.exterior.coords)[:3]], dtype=np.float64)
                    tris = np.arange(len(pts)).reshape(-1, 3)

                a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
                cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
                tris = np.where((cross < 0)[:, None], tris[:, [0, 2, 1]], tris)
                tris = tris[np.abs(cross) > 1e-9]
                if len(tris) == 0: continue

                v = np.zeros((len(pts), 15), np.float32)
                v[:, 0:2] = pts; v[:, 2] = z
                v[:, 3:6] = (0, 0, 1); v[:, 6:9] = color; v[:, 9] = 1.0
                verts_list.append(v)
                idx_list.append(tris + offset)
                offset += len(pts)

        if not verts_list: return None
        V = np.vstack(verts_list)
        I = np.ascontiguousarray(np.vstack(idx_list).ravel(), dtype=np.uint32)
        return MeshData(V, I, np.empty((0, 6), np.float32), float(np.max(np.hypot(V[:, 0], V[:, 1]))), 0.1)

class MeshCache:
    def __init__(self, cache_dir, enabled=True):
        self.enabled = enabled
        self.dir = Path(cache_dir) / "mesh"

    def path(self, z, key):
        # v2_ prefix forcefully invalidates any previous cached meshes
        return self.dir / f"v2_{MESH_CACHE_VERSION}_{z}_{key[0]}_{key[1]}.npz"

    def load(self, z, key):
        if not self.enabled: return None
        p = self.path(z, key)
        if not p.exists(): return None
        try:
            with np.load(p, allow_pickle=False) as d:
                b_mesh = MeshData(d["b_verts"], d["b_indices"], d["b_edges"], float(d["b_radius"]), float(d["b_height"])) if "b_verts" in d and len(d["b_verts"]) > 0 else None
                g_mesh = MeshData(d["g_verts"], d["g_indices"], np.empty((0, 6), np.float32), float(d["g_radius"]), 0.0) if "g_verts" in d and len(d["g_verts"]) > 0 else None
                w_mesh = MeshData(d["w_verts"], d["w_indices"], np.empty((0, 6), np.float32), float(d["w_radius"]), 0.0) if "w_verts" in d and len(d["w_verts"]) > 0 else None
                t_arr = d["trees"] if "trees" in d else np.empty((0, 8), np.float32)
        except Exception:
            try: p.unlink()
            except OSError: pass
            return None
        return ChunkResult(tuple(key), b_mesh, g_mesh, w_mesh, t_arr if len(t_arr) else None)

    def store(self, z, key, b_mesh, g_mesh, w_mesh, t_mesh):
        if not self.enabled: return
        payload = {}
        if b_mesh: payload.update(b_verts=b_mesh.vertices, b_indices=b_mesh.indices, b_edges=b_mesh.edges, b_radius=b_mesh.radius, b_height=b_mesh.max_height)
        if g_mesh: payload.update(g_verts=g_mesh.vertices, g_indices=g_mesh.indices, g_radius=g_mesh.radius)
        if w_mesh: payload.update(w_verts=w_mesh.vertices, w_indices=w_mesh.indices, w_radius=w_mesh.radius)
        if t_mesh is not None: payload["trees"] = t_mesh
        p = self.path(z, key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
            with open(tmp, "wb") as f: np.savez_compressed(f, **payload)
            os.replace(tmp, p)
        except OSError as exc: print(f"[mesh] cache write failed: {exc}")

def mesh_tile_job(payload):
    raw, lat0, lon0, z, x, y, default_height, level_height = payload
    proj = LocalProjection(lat0, lon0)
    buildings, roads, greens, waters, trees = parse_tile_elements(raw, proj, (z, x, y), default_height, level_height)
    b_mesh, g_mesh, w_mesh, t_mesh = MeshExtruder().build(buildings, roads, greens, waters, trees)
    
    def unpack(m):
        if m is None or not np.isfinite(m.vertices).all() or not np.isfinite(m.edges).all(): return None
        if np.max(np.hypot(m.vertices[:, 0], m.vertices[:, 1])) > 10000.0: return None
        idx = np.ascontiguousarray(m.indices, dtype=np.uint32)
        if len(idx) == 0 or len(idx) % 3 or int(idx.max()) >= len(m.vertices): return None
        return (m.vertices, idx, m.edges, m.radius, m.max_height)
        
    if t_mesh is not None and (not np.isfinite(t_mesh).all() or np.max(np.hypot(t_mesh[:, 0], t_mesh[:, 1])) > 10000.0):
        t_mesh = None
    return unpack(b_mesh), unpack(g_mesh), unpack(w_mesh), t_mesh

class MeshPool:
    def __init__(self, workers, use_processes=True):
        self.workers = max(1, int(workers))
        if sys.platform == "win32":
            use_processes = False  
        self.executor = ProcessPoolExecutor(max_workers=self.workers) if use_processes else ThreadPoolExecutor(max_workers=self.workers)
    def submit(self, payload): return self.executor.submit(mesh_tile_job, payload)
    def shutdown(self): self.executor.shutdown(wait=False, cancel_futures=True)

_mesh_pool = None
_mesh_pool_lock = threading.Lock()

def shared_mesh_pool(workers, use_processes=True):
    global _mesh_pool
    with _mesh_pool_lock:
        if _mesh_pool is None: _mesh_pool = MeshPool(workers, use_processes)
        return _mesh_pool

def shutdown_mesh_pool():
    global _mesh_pool
    with _mesh_pool_lock:
        pool, _mesh_pool = _mesh_pool, None
    if pool: pool.shutdown()