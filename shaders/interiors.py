"""Finite 3D rooms: ray-hit furniture, filtered materials and practical lighting."""
from .interior_layouts import INTERIOR_LAYOUTS


INTERIORS = """
// Room coordinates are metres: X along the facade, Y up, Z into the building.
// Only the facade writes scene depth; the nearest solid inside it determines
// visibility. Furniture never inherits a panorama's perspective or curvature.
const int IR_WOOD = 1, IR_METAL = 2, IR_FABRIC = 3, IR_BOOKS = 4;
const int IR_SCREEN = 5, IR_CABINET = 6, IR_LINEN = 7, IR_PILLOW = 8;
const int IR_POT = 9, IR_LEAVES = 10, IR_SHADE = 11, IR_STOCK = 12;
const int IR_PLASTER = 13;

float interiorExposureBlend() {
    return clamp(log2(max(uExposure / DAY_EXPOSURE, 1.0)) / NIGHT_EXPOSURE_STOPS, 0.0, 1.0);
}
float interiorSurfaceGain() {
    return nits(OFFICE_ILLUMINANCE / PI)
         * exp2(mix(log2(INTERIOR_DAY_GAIN), log2(INTERIOR_NIGHT_GAIN), interiorExposureBlend()));
}
float interiorFixtureGain() {
    return exp2(mix(log2(INTERIOR_FIXTURE_DAY_GAIN),
                    log2(INTERIOR_FIXTURE_NIGHT_GAIN), interiorExposureBlend()));
}
vec3 interiorLight(vec3 albedo, float diffuse, float fixture, float lit,
                   vec3 lampCol, vec3 ambient) {
    vec3 lamp = lampCol / max(dot(lampCol, PHOTO_LUMA), 1e-5);
    return albedo * (ambient + lamp * lit * interiorSurfaceGain() * diffuse)
         + emissionNits(mix(vec3(1.0), lampCol, .65), CEILING_NITS)
         * fixture * lit * interiorFixtureGain();
}

struct RoomRay {
    vec3 o, d, inv;
    float exitT;
    vec3 exitN, floorPoint;
};
struct RoomHit {
    float t;
    vec3 n, center, halfSize;
    int material, shape;
    float contact;
};

RoomRay interiorRay(vec3 d, vec2 o, vec3 size, vec4 h) {
    // Mirroring is a rigid layout variation shared by every pane in the suite.
    if (h.z < .5) { o.x = size.x - o.x; d.x = -d.x; }
    d.z = max(d.z, .00001);
    vec3 ro = vec3(clamp(o, vec2(.001), size.xy - .001), 0.0);
    vec3 inv = 1.0 / mix(mix(vec3(-.000001), vec3(.000001), step(vec3(0.0), d)),
                         d, step(vec3(.000001), abs(d)));
    vec3 exits = (mix(vec3(0.0), size, step(vec3(0.0), d)) - ro) * inv;
    float t = min(exits.x, min(exits.y, exits.z));
    vec3 n = exits.z <= min(exits.x, exits.y) ? vec3(0.0, 0.0, -1.0)
           : exits.y <= exits.x ? vec3(0.0, -sign(d.y), 0.0) : vec3(-sign(d.x), 0.0, 0.0);
    return RoomRay(ro, d, inv, t, n, ro + d * t);
}

void roomContact(RoomRay ray, vec3 c, vec3 s, int material, inout RoomHit hit) {
    if (ray.exitN.y < .5 || material == IR_SHADE || material == IR_LEAVES) return;
    vec2 outside = max(abs(ray.floorPoint.xz - c.xz) - s.xz, vec2(0.0));
    float height = max(c.y - s.y, 0.0);
    float shadow = (1.0 - smoothstep(.0, .18 + height * .28, length(outside)))
                 * exp(-height * .55);
    hit.contact = max(hit.contact, shadow);
}

void roomBox(RoomRay ray, vec3 c, vec3 s, int material, inout RoomHit hit) {
    vec3 a = (c - s - ray.o) * ray.inv, b = (c + s - ray.o) * ray.inv;
    vec3 lo = min(a, b), hi = max(a, b);
    float nearT = max(lo.x, max(lo.y, lo.z));
    float farT = min(hi.x, min(hi.y, hi.z));
    roomContact(ray, c, s, material, hit);
    if (nearT <= 0.0 || nearT > farT || nearT >= hit.t) return;
    vec3 n = lo.x >= max(lo.y, lo.z) ? vec3(-sign(ray.d.x), 0.0, 0.0)
           : lo.y >= lo.z ? vec3(0.0, -sign(ray.d.y), 0.0) : vec3(0.0, 0.0, -sign(ray.d.z));
    hit.t = nearT; hit.n = n; hit.center = c; hit.halfSize = s;
    hit.material = material; hit.shape = 0;
}

void roomSoftBox(RoomRay ray, vec3 c, vec3 s, float radius, int material, inout RoomHit hit) {
    // Tight box rejection first; only rays at a cushion's rounded edge need steps.
    RoomHit candidate = hit;
    roomBox(ray, c, s, material, candidate);
    hit.contact = candidate.contact;
    if (candidate.t >= hit.t) return;
    float t = candidate.t, speed = length(ray.d);
    float r = min(radius, min(s.x, min(s.y, s.z)) * .85);
    for (int i = 0; i < 12; ++i) {
        vec3 q = abs(ray.o + ray.d * t - c) - (s - r);
        float distance = length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
        if (distance < .0005) {
            hit.t = t; hit.center = c; hit.halfSize = s;
            hit.n = normalize(max(q, vec3(0.0)) * sign(ray.o + ray.d * t - c));
            hit.material = material; hit.shape = 1;
            return;
        }
        t += distance / speed;
        if (t >= hit.t) return;
    }
}

void roomEllipsoid(RoomRay ray, vec3 c, vec3 s, int material, inout RoomHit hit) {
    vec3 o = (ray.o - c) / s, d = ray.d / s;
    float a = dot(d, d), b = dot(o, d), disc = b * b - a * (dot(o, o) - 1.0);
    if (disc < 0.0) return;
    float t = (-b - sqrt(disc)) / a;
    if (t <= 0.0 || t >= hit.t) return;
    hit.t = t; hit.n = normalize((ray.o + ray.d * t - c) / (s * s));
    hit.center = c; hit.halfSize = s; hit.material = material; hit.shape = 1;
}

void roomCylinder(RoomRay ray, vec3 c, float radius, float halfHeight,
                   int material, inout RoomHit hit) {
    vec3 o = ray.o - c;
    float a = dot(ray.d.xz, ray.d.xz), b = dot(o.xz, ray.d.xz);
    float disc = b * b - a * (dot(o.xz, o.xz) - radius * radius);
    float t = 1e8; vec3 n = vec3(0.0);
    if (disc >= 0.0 && a > 1e-8) {
        float side = (-b - sqrt(disc)) / a;
        if (side > 0.0 && abs(o.y + side * ray.d.y) <= halfHeight) {
            t = side; n = normalize(vec3(o.x + t * ray.d.x, 0.0, o.z + t * ray.d.z));
        }
    }
    float capY = ray.d.y > 0.0 ? -halfHeight : halfHeight;
    float cap = (capY - o.y) * ray.inv.y;
    vec2 at = o.xz + ray.d.xz * cap;
    if (cap > 0.0 && cap < t && dot(at, at) <= radius * radius) {
        t = cap; n = vec3(0.0, sign(capY), 0.0);
    }
    vec3 s = vec3(radius, halfHeight, radius);
    roomContact(ray, c, s, material, hit);
    if (t >= hit.t) return;
    hit.t = t; hit.n = n; hit.center = c; hit.halfSize = s;
    hit.material = material; hit.shape = 2;
}

int interiorVariant(int kind, vec4 h) {
    return int(floor(h.y * (kind >= 5 ? 6.0 : 3.0)));
}
void interiorLamps(vec3 size, int kind, vec4 h, out vec3 a, out vec3 b) {
    float drop = kind == 0 || kind == 6 ? .62 : .055;
    a = vec3(size.x * (.28 + .08 * h.x), size.y - drop, size.z * (.30 + .08 * h.w));
    b = vec3(size.x * (.68 + .08 * h.w), size.y - drop, size.z * (.63 + .08 * h.x));
}
""" + INTERIOR_LAYOUTS + """

RoomHit traceInterior(RoomRay ray, vec3 size, int kind, vec4 h) {
    RoomHit hit = RoomHit(ray.exitT, ray.exitN, size * .5, size * .5, 0, 0, 0.0);
    interiorFurniture(ray, size, kind, h, hit);
    return hit;
}

vec3 interiorWall(vec4 h) {
    if (h.x < .22) return vec3(.81, .79, .72);
    if (h.x < .43) return vec3(.74, .77, .75);
    if (h.x < .64) return vec3(.78, .74, .68);
    if (h.x < .83) return vec3(.69, .74, .76);
    return vec3(.85, .83, .77);
}
vec3 interiorWood(vec4 h) {
    return h.w < .34 ? vec3(.48, .34, .23) : h.w < .70 ? vec3(.67, .52, .35) : vec3(.34, .26, .22);
}
vec3 interiorCloth(vec4 h) {
    float k = fract(h.x + h.w * .73);
    if (k < .23) return vec3(.43, .49, .37);
    if (k < .45) return vec3(.38, .47, .49);
    if (k < .64) return vec3(.55, .38, .29);
    if (k < .84) return vec3(.51, .48, .43);
    return vec3(.35, .36, .40);
}
float roomRect(vec2 p, vec2 lo, vec2 hi, float fw) {
    vec2 a = smoothstep(lo - fw, lo + fw, p) * (1.0 - smoothstep(hi - fw, hi + fw, p));
    return a.x * a.y;
}

vec3 roomArtwork(vec2 uv, vec4 h, float index, float fw) {
    vec3 paper = vec3(.79, .76, .66);
    vec3 ink = mix(vec3(.31, .42, .39), vec3(.57, .34, .23), fract(h.w + index * .37));
    float style = hash21(vec2(index + h.x * 31.0, h.y * 17.0));
    fw = max(fw, .008);
    if (style < .28) {
        float disc = 1.0 - smoothstep(.22 - fw, .22 + fw, length(uv - vec2(.40 + .15 * h.z, .63)));
        float hill = smoothstep(-fw, fw, .29 + .12 * sin(uv.x * 4.0 + index) - uv.y);
        return mix(mix(paper, ink, disc * .75), ink * .65, hill);
    }
    if (style < .52) {
        float a = roomRect(uv, vec2(.14, .18), vec2(.45, .82), fw);
        float b = roomRect(uv, vec2(.52, .31), vec2(.84, .71), fw);
        return mix(mix(paper, ink, a), mix(ink, vec3(.64, .53, .33), .7), b);
    }
    if (style < .81) {
        float stem = 1.0 - smoothstep(.007, .007 + fw, abs(uv.x - .49 - .04 * sin(uv.y * 3.0)));
        stem *= roomRect(uv, vec2(.0, .14), vec2(1.0, .86), fw);
        float leaves = 0.0;
        for (int i = 0; i < 5; ++i) {
            float side = mod(float(i), 2.0) * 2.0 - 1.0;
            vec2 at = vec2(.51 + side * .10, .27 + float(i) * .115);
            vec2 q = (uv - at) / vec2(.13, .046);
            leaves = max(leaves, 1.0 - smoothstep(.83, 1.0 + fw * 8.0, length(q)));
        }
        return mix(paper, ink * .65, max(stem, leaves) * .85);
    }
    float hills = uv.y - (.32 + .17 * sin(uv.x * 4.1 + index));
    float line = 1.0 - smoothstep(.012, .012 + fw, abs(hills));
    float water = roomRect(uv, vec2(.13, .16), vec2(.86, .27), fw)
                * band(uv.y * 32.0, .2, .5, fw * 32.0);
    return mix(paper, ink * .70, max(line, water) * .85);
}

vec3 interiorShell(vec3 P, vec3 N, vec3 size, int kind, vec4 h, float fw,
                    out float fixture) {
    vec3 wall = interiorWall(h), wood = interiorWood(h), alb = wall;
    fixture = 0.0;
    float fine = 1.0 - smoothstep(.035, .16, fw);
    bool office = kind == 5;
    if (N.y > .5) {
        if (office && h.w < .70) {
            vec2 id = floor(P.xz / .60);
            float tile = mix(1.0, .90 + .15 * hash21(id + h.z * 19.0), fine);
            float seam = band(P.x / .60, .012, .988, fw / .60) * band(P.z / .60, .012, .988, fw / .60);
            alb = mix(vec3(.31, .33, .32), vec3(.39, .36, .31), h.x) * tile * (.9 + .1 * seam);
        } else if (kind == 2 || kind == 3 || kind == 4) {
            float sz = kind == 3 ? 1.2 : .65;
            float seam = band(P.x / sz, .012, .988, fw / sz) * band(P.z / sz, .012, .988, fw / sz);
            float check = mod(floor(P.x / sz) + floor(P.z / sz), 2.0);
            alb = mix(vec3(.60, .58, .51), vec3(.45, .48, .46), .12 + check * .12 * fine) * (.84 + .16 * seam);
        } else {
            float row = floor(P.x / .19);
            float board = hash21(vec2(row, floor((P.z + row * .53) / 1.65)) + h.x * 9.0);
            float seam = band(P.x / .19, .012, .988, fw / .19)
                       * band((P.z + row * .53) / 1.65, .003, .997, fw / 1.65);
            vec3 timber = kind == 0 ? mix(wood, vec3(.65, .59, .49), .60) : wood;
            float grain = sin(P.x * 175.0 + sin(P.z * 1.4 + row) * 1.8);
            float grainFade = 1.0 - smoothstep(.005, .035, fw);
            alb = timber * mix(1.0, .91 + .17 * board, fine) * (.94 + .06 * seam)
                  * (1.0 + .025 * grain * grainFade);
        }
        // Woven rugs anchor seating and beds; their detail is flat on the floor.
        if (kind == 6 || (kind == 0 && interiorVariant(kind, h) == 0)) {
            vec2 halfRug = size.xz * vec2(.32, .28);
            vec2 at = P.xz - size.xz * vec2(.49, .45);
            float rug = roomRect(at, -halfRug, halfRug, fw);
            float border = roomRect(at, -halfRug + .12, halfRug - .12, fw);
            vec3 cloth = mix(interiorCloth(h) * .72, mix(interiorCloth(h), vec3(.70, .64, .52), .45), border);
            alb = mix(alb, cloth, rug * .90);
        }
    } else if (N.y < -.5) {
        alb = vec3(.86, .85, .81);
        if (office || kind == 2 || kind == 3) {
            float grid = band(P.x / .60, .009, .991, fw / .60) * band(P.z / .60, .009, .991, fw / .60);
            alb *= .89 + .11 * grid;
        } else if (kind != 0) {
            float beam = 1.0 - band(P.z / 2.15 + h.w, .035, .965, fw / 2.15);
            alb = mix(alb, wall * .78, beam * .55);
        }
        if (kind != 0 && kind != 6) {
            vec3 a, b; interiorLamps(size, kind, h, a, b);
            vec2 hs = h.z < .65 ? vec2(.29, .58) : vec2(.10, .94);
            float trim = max(roomRect(P.xz - a.xz, -hs - .04, hs + .04, fw),
                             roomRect(P.xz - b.xz, -hs - .04, hs + .04, fw));
            fixture = max(roomRect(P.xz - a.xz, -hs, hs, fw), roomRect(P.xz - b.xz, -hs, hs, fw));
            alb *= 1.0 - .25 * trim;
        }
    } else {
        vec2 p = abs(N.x) > .5 ? P.zy : P.xy;
        float width = abs(N.x) > .5 ? size.z : size.x;
        bool accent = kind == 0 || (kind == 1 && h.x < .5);
        if (accent && (N.x < -.5 || N.z < -.5)) {
            vec2 brick = p / vec2(.28, .085);
            brick.x += .5 * mod(floor(brick.y), 2.0);
            float inside = band(brick.x, .026, .974, fw / .28) * band(brick.y, .07, .93, fw / .085);
            vec3 clay = mix(vec3(.58, .42, .31), vec3(.73, .66, .52), h.x);
            clay *= mix(1.0, .92 + .15 * hash21(floor(brick)), fine);
            alb = mix(vec3(.66, .63, .54), clay, inside);
        }
        if (P.y < .13) alb = wood * .56;
        float cornice = smoothstep(size.y - .14 - fw, size.y - .14 + fw, P.y);
        alb = mix(alb, wall * .85, cornice);
        // Back doors, elevator panels, artwork and wall trim are planar details.
        float dx = width * mix(.20, .78, step(.5, h.w));
        if (N.z < -.5) {
            float door = roomRect(p, vec2(dx - .45, .13), vec2(dx + .45, 2.25), fw);
            float inner = roomRect(p, vec2(dx - .40, .14), vec2(dx + .40, 2.19), fw);
            vec3 doorCol = kind == 4 ? vec3(.53, .55, .53) : wood * .77;
            alb = mix(alb, mix(vec3(.16), doorCol, inner), door);
            float handle = roomRect(p, vec2(dx + .26, 1.00), vec2(dx + .32, 1.14), fw);
            alb = mix(alb, vec3(.58, .58, .53), handle * door);
        }
        if (kind == 3 && abs(N.x) > .5) {
            vec2 bay = vec2(mod(p.x, 3.2), p.y);
            float recess = roomRect(bay, vec2(.18, .18), vec2(2.98, 2.7), fw);
            float pane = roomRect(bay, vec2(.24, .25), vec2(2.92, 2.61), fw);
            alb = mix(alb, mix(vec3(.13, .15, .15), interiorCloth(h) * .70, pane), recess);
        } else {
            float spacing = office ? width : 2.5;
            float index = floor((p.x + h.z * .7) / spacing);
            vec2 art = vec2(mod(p.x + h.z * .7, spacing) - spacing * .5, p.y - 1.73);
            vec2 hs = office ? vec2(.75, .43) : vec2(.32, .40);
            float present = step(.22, hash21(vec2(index + h.x * 5.0, N.x + h.y * 8.0)));
            if (kind == 0 && abs(N.x) > .5) {
                // A compact gallery of six prints, rather than a repeated wall stripe.
                float col = floor((p.x - width * .32) / .85);
                float row = step(1.73, p.y);
                art = vec2(mod(p.x - width * .32, .85) - .425, p.y - (1.33 + row * .80));
                hs = vec2(.29, .32);
                index = col + row * 3.0 + N.x * 17.0;
                present = step(0.0, col) * step(col, 2.0);
            }
            if (N.z < -.5) present *= step(.52 + hs.x, abs((p.x - art.x) - dx));
            float frame = roomRect(art, -hs, hs, fw);
            float paper = roomRect(art, -hs + .035, hs - .035, fw);
            if (frame * present > .001) {
                vec3 picture = roomArtwork(art / (hs * 2.0) + .5, h, index, fw / (hs.x * 2.0));
                if (office) {
                    float notes = band((art.y + .3) * 8.0, .05, .10, fw * 8.0)
                                * roomRect(art, vec2(-.55, -.20), vec2(.2, .25), fw);
                    picture = mix(vec3(.77, .80, .77), vec3(.42, .48, .46), notes);
                    picture = mix(picture, vec3(.73, .65, .40),
                                  roomRect(art, vec2(.37, -.16), vec2(.54, .02), fw));
                }
                alb = mix(alb, mix(wood * .38, picture, paper), frame * present);
            }
        }
    }
    return linearColor(alb);
}

vec3 interiorObject(RoomHit hit, vec3 P, vec4 h, float fw, out float fixture) {
    vec3 q = P - hit.center, s = hit.halfSize;
    vec3 alb = interiorWood(h);
    fixture = 0.0;
    float fine = 1.0 - smoothstep(.025, .12, fw);
    int m = hit.material;
    if (m == IR_METAL) alb = vec3(.17, .18, .18);
    else if (m == IR_FABRIC) alb = interiorCloth(h);
    else if (m == IR_LINEN) alb = mix(vec3(.80, .78, .70), interiorCloth(h), .24);
    else if (m == IR_PILLOW) alb = mix(vec3(.87, .84, .76), interiorCloth(h), step(.52, h.w) * .6);
    else if (m == IR_POT) alb = h.x < .5 ? vec3(.55, .39, .28) : vec3(.64, .64, .57);
    else if (m == IR_LEAVES) {
        float leaves = .94 + .06 * sin(P.y * 23.0 + sin(P.x * 17.0) * 2.0) * fine;
        alb = vec3(.33, .43, .27) * leaves;
    } else if (m == IR_PLASTER) alb = interiorWall(h);
    else if (m == IR_SHADE) {
        float cage = band(atan(q.z, q.x) * 3.18, .10, .87, fw / max(s.x, .05) * 3.18);
        float rim = smoothstep(s.y - .035 - fw, s.y - .035 + fw, abs(q.y));
        alb = mix(vec3(.18, .17, .13), vec3(.84, .78, .59), cage * (1.0 - rim));
        fixture = (hit.n.y < -.5 ? .9 : .58 * cage) * (1.0 - rim * .8);
    } else if (m == IR_BOOKS || m == IR_STOCK || m == IR_CABINET || m == IR_SCREEN) {
        bool alongX = s.z <= s.x;
        vec2 face = alongX ? q.xy : q.zy;
        vec2 hs = alongX ? s.xy : s.zy;
        float facing = alongX ? abs(hit.n.z) : abs(hit.n.x);
        vec2 uv = face + hs;
        float inset = roomRect(face, -hs + .045, hs - .045, fw) * facing;
        if (m == IR_SCREEN) {
            vec2 p = face / (hs * 2.0) + .5;
            float content = roomRect(p, vec2(.09, .15), vec2(.58, .87), fw / max(hs.x, .1));
            float line = band(p.y * 8.0, .12, .63, fw * 8.0) * step(.62, p.x) * step(p.x, .90);
            alb = mix(vec3(.045, .05, .05), vec3(.20, .31, .34), inset);
            fixture = inset * (.045 + content * .095 + line * .055);
        } else if (m == IR_CABINET) {
            float seams = band(uv.x / .55, .008, .992, fw / .55) * band(uv.y / .32, .008, .992, fw / .32);
            alb = mix(interiorWood(h) * .72, interiorWall(h) * .89, step(.5, h.x)) * (.74 + .26 * seams);
        } else {
            float pitch = m == IR_BOOKS ? .105 : .16;
            vec2 id = floor(uv / vec2(pitch, .39));
            vec2 cell = fract(uv / vec2(pitch, .39));
            float r = hash21(id + hit.center.xz + h.w * 13.0);
            float stock = band(uv.x / pitch, .07, .92, fw / pitch)
                        * band(uv.y / .39, .12, mix(.62, .93, r), fw / .39);
            vec3 color = mix(interiorCloth(h), mix(vec3(.71, .55, .35), vec3(.50, .57, .57), r), .62);
            float label = roomRect(cell, vec2(.2, .30), vec2(.78, .42), fw / pitch);
            color = mix(color, vec3(.78, .75, .65), label * .6);
            vec3 contents = mix(interiorWood(h) * .23, color, mix(.65, stock, fine));
            alb = mix(alb, contents, inset);
        }
    }
    if (m == IR_WOOD || m == IR_CABINET) {
        float grain = sin(P.x * 19.0 + sin(P.z * 3.4) * 1.2);
        alb *= 1.0 + .035 * grain * fine;
    }
    return linearColor(alb);
}

vec3 furnishedRoom(vec3 d, vec2 o, vec2 size, float D, int kind,
                    vec4 h, float lit, vec3 lampCol, vec3 amb, vec2 footprint) {
    vec3 rs = vec3(max(size, vec2(.5)), max(D, 2.0));
    float dist = length(uCamPos - vWorld);
    float pixel = max(footprint.x, footprint.y);
    float lod = max(smoothstep(140.0, 300.0, dist),
                    smoothstep(.14, .55, max(footprint.x / rs.x, footprint.y / rs.y)));
    vec3 mean = linearColor(interiorWall(h)) * .49 + linearColor(interiorWood(h)) * .19
              + linearColor(vec3(.84, .83, .79)) * .23;
    vec3 flatRad = interiorLight(mean, .74, .011, lit, lampCol, amb * .45);
    if (lod >= .999) return flatRad;

    RoomRay ray = interiorRay(d, o, rs, h);
    RoomHit hit = traceInterior(ray, rs, kind, h);
    vec3 P = ray.o + ray.d * hit.t, N = hit.n;
    float fw = max(.001, pixel * (1.0 + hit.t / max(dist, .5)));
    float fixture;
    vec3 albedo = hit.material == 0 ? interiorShell(P, N, rs, kind, h, fw, fixture)
                                    : interiorObject(hit, P, h, fw, fixture);
    if (hit.material > 0 && hit.shape == 0) {
        float bevel = min(.035, min(hit.halfSize.x, min(hit.halfSize.y, hit.halfSize.z)) * .28);
        if (hit.material == IR_FABRIC || hit.material == IR_LINEN || hit.material == IR_PILLOW) bevel *= 2.5;
        vec3 edge = max(abs(P - hit.center) - (hit.halfSize - bevel), vec3(0.0));
        N = normalize(edge * sign(P - hit.center) + hit.n * .00001);
    }
    float junction;
    if (hit.material != 0) junction = max(P.y, .0);
    else if (abs(N.y) > .5) junction = min(min(P.x, rs.x - P.x), rs.z - P.z);
    else if (abs(N.z) > .5) junction = min(min(P.x, rs.x - P.x), min(P.y, rs.y - P.y));
    else junction = min(min(P.y, rs.y - P.y), rs.z - P.z);
    float ao = .64 + .36 * smoothstep(0.0, .48, junction);
    if (hit.material == 0 && N.y > .5) ao *= 1.0 - .42 * hit.contact;
    if (hit.material > 0 && N.y < .1) ao *= .83 + .17 * smoothstep(.05, .65, P.y);

    vec3 a, b; interiorLamps(rs, kind, h, a, b);
    vec3 va = a - P, vb = b - P;
    float aa = dot(va, va), bb = dot(vb, vb);
    float direct = max(dot(N, va * inversesqrt(max(aa, .0001))), 0.0) / (1.0 + .22 * aa)
                 + max(dot(N, vb * inversesqrt(max(bb, .0001))), 0.0) / (1.0 + .22 * bb);
    float bounce = mix(.38, .68, max(-N.y, 0.0)) * (1.0 - .18 * P.z / rs.z);
    float diffuse = (direct * 1.65 + bounce) * ao;
    float windowLight = (.32 + 1.05 * exp(-P.z / 3.8)) * (.45 + .55 * max(-N.z, 0.0));
    vec3 rad = interiorLight(albedo, diffuse, fixture, lit, lampCol, amb * windowLight * ao);
    return mix(rad, flatRad, lod);
}

vec3 interiorRadiance(vec3 d, vec2 o, vec2 rs, float D, vec4 h, float lit,
                       vec3 lampCol, vec3 amb, bool office, vec2 footprint) {
    return furnishedRoom(d, o, rs, D, office ? 5 : 6, h, lit, lampCol, amb, footprint);
}

vec3 windowDressing(vec3 rad, vec3 d, vec2 pane, vec2 baySize, vec4 rect,
                     vec4 h, float lit, vec3 lampCol, vec3 amb, bool office) {
    vec2 p = (pane + d.xy / max(d.z, .04) * .18 / baySize - rect.xz)
           / max(rect.yw - rect.xz, vec2(.05));
    vec2 fw = max(fwidth(p), vec2(.001));
    vec3 lamp = lampCol / max(dot(lampCol, PHOTO_LUMA), 1e-5);
    vec3 fill = amb * 1.3 + lamp * lit * interiorSurfaceGain() * .62;
    if (office) {
        float present = step(.70, h.w);
        float bottom = 1.0 - mix(.10, .55, h.z);
        float cover = smoothstep(bottom - fw.y, bottom + fw.y, p.y) * present;
        float slat = band(p.y * baySize.y / .055, .12, .84, fw.y * baySize.y / .055);
        float roller = step(.86, h.w);
        vec3 cloth = linearColor(vec3(.74, .735, .69)) * mix(.72 + .28 * slat, .90, roller);
        return mix(rad, cloth * fill, cover * .92);
    }
    float present = step(.18, h.w);
    float width = mix(.08, .23, h.x);
    width = mix(width, .48, step(.94, h.z));
    float rightWidth = width * mix(.65, 1.30, h.y);
    float cover = max(1.0 - smoothstep(width - fw.x, width + fw.x, p.x),
                     smoothstep(1.0 - rightWidth - fw.x, 1.0 - rightWidth + fw.x, p.x)) * present;
    cover *= smoothstep(.035 - fw.y, .035 + fw.y, p.y);
    float detail = 1.0 - smoothstep(.008, .05, fw.x);
    float phase = p.x * baySize.x * (rect.y - rect.x) * 48.0 + .3 * sin(p.y * 2.6 + h.z * 5.0);
    float folds = .83 + (.10 * cos(phase) + .025 * cos(phase * 2.0 + .4)) * detail;
    vec3 cloth = linearColor(mix(vec3(.82, .78, .68), vec3(.61, .65, .64), h.x));
    return mix(rad, cloth * fill * folds, cover * .94);
}
"""
