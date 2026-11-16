"""Seeded, metre-scale furniture assemblies for the analytic 3D room shader."""

INTERIOR_LAYOUTS = """
void roomChair(RoomRay ray, vec2 at, vec2 back, bool soft, inout RoomHit hit) {
    int material = soft ? IR_FABRIC : IR_WOOD;
    roomSoftBox(ray, vec3(at.x, .48, at.y), vec3(.245, .065, .245), .045, material, hit);
    roomSoftBox(ray, vec3(at.x + back.x * .235, .79, at.y + back.y * .235),
                vec3(.245 - .20 * abs(back.x), .26, .245 - .20 * abs(back.y)), .035, material, hit);
    if (soft) {
        roomCylinder(ray, vec3(at.x, .235, at.y), .045, .205, IR_METAL, hit);
        roomBox(ray, vec3(at.x, .035, at.y), vec3(.24, .025, .035), IR_METAL, hit);
        roomBox(ray, vec3(at.x, .035, at.y), vec3(.035, .025, .24), IR_METAL, hit);
    } else {
        for (int i = 0; i < 4; ++i) {
            vec2 leg = vec2(mod(float(i), 2.0), floor(float(i) / 2.0)) * .38 - .19;
            roomBox(ray, vec3(at.x + leg.x, .21, at.y + leg.y), vec3(.024, .21, .024), IR_WOOD, hit);
        }
    }
}

void roomDesk(RoomRay ray, vec2 at, float width, bool chair, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .77, at.y), vec3(width * .5, .035, .38), IR_WOOD, hit);
    roomBox(ray, vec3(at.x - width * .37, .38, at.y), vec3(.035, .36, .32), IR_METAL, hit);
    roomBox(ray, vec3(at.x + width * .37, .38, at.y), vec3(.18, .36, .32), IR_CABINET, hit);
    roomBox(ray, vec3(at.x, 1.15, at.y + .06), vec3(.32, .205, .04), IR_SCREEN, hit);
    roomBox(ray, vec3(at.x, .89, at.y + .06), vec3(.045, .085, .035), IR_METAL, hit);
    roomBox(ray, vec3(at.x, .817, at.y - .21), vec3(.23, .012, .075), IR_METAL, hit);
    roomBox(ray, vec3(at.x + width * .35, .818, at.y - .07), vec3(.105, .015, .14), IR_PILLOW, hit);
    if (chair) roomChair(ray, at + vec2(.07, .79), vec2(0.0, 1.0), true, hit);
}

void roomTable(RoomRay ray, vec2 at, vec2 halfSize, float height, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, height, at.y), vec3(halfSize.x, .035, halfSize.y), IR_WOOD, hit);
    roomBox(ray, vec3(at.x - halfSize.x * .73, height * .5, at.y),
            vec3(.045, height * .5 - .03, halfSize.y * .78), IR_METAL, hit);
    roomBox(ray, vec3(at.x + halfSize.x * .73, height * .5, at.y),
            vec3(.045, height * .5 - .03, halfSize.y * .78), IR_METAL, hit);
}

void roomRoundTable(RoomRay ray, vec2 at, float radius, inout RoomHit hit) {
    roomCylinder(ray, vec3(at.x, .75, at.y), radius, .035, IR_WOOD, hit);
    roomCylinder(ray, vec3(at.x, .365, at.y), .055, .355, IR_METAL, hit);
    roomCylinder(ray, vec3(at.x, .035, at.y), radius * .47, .025, IR_METAL, hit);
}

void roomSofa(RoomRay ray, vec2 at, float width, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .18, at.y), vec3(width * .48, .10, .40), IR_WOOD, hit);
    roomSoftBox(ray, vec3(at.x, .72, at.y + .34), vec3(width * .50, .31, .12), .08, IR_FABRIC, hit);
    roomSoftBox(ray, vec3(at.x - width * .47, .50, at.y), vec3(.115, .27, .45), .075, IR_FABRIC, hit);
    roomSoftBox(ray, vec3(at.x + width * .47, .50, at.y), vec3(.115, .27, .45), .075, IR_FABRIC, hit);
    for (int i = 0; i < 3; ++i) {
        float x = at.x + (float(i) - 1.0) * width * .28;
        roomSoftBox(ray, vec3(x, .405, at.y - .015), vec3(width * .134, .12, .35), .08, IR_FABRIC, hit);
    }
    roomSoftBox(ray, vec3(at.x - width * .31, .66, at.y + .16), vec3(.22, .22, .075), .06, IR_PILLOW, hit);
    roomSoftBox(ray, vec3(at.x + width * .30, .66, at.y + .16), vec3(.22, .19, .075), .06, IR_PILLOW, hit);
}

void roomPlant(RoomRay ray, vec2 at, float scale, inout RoomHit hit) {
    roomCylinder(ray, vec3(at.x, .23 * scale, at.y), .20 * scale, .23 * scale, IR_POT, hit);
    roomBox(ray, vec3(at.x, .69 * scale, at.y), vec3(.025, .47 * scale, .025), IR_WOOD, hit);
    for (int i = 0; i < 7; ++i) {
        float angle = float(i) * 2.40;
        vec2 offset = vec2(cos(angle), sin(angle)) * .17 * scale;
        roomEllipsoid(ray, vec3(at.x + offset.x, (.78 + float(i) * .105) * scale, at.y + offset.y),
                      vec3(.22, .065, .15) * scale, IR_LEAVES, hit);
    }
    roomEllipsoid(ray, vec3(at.x, 1.49 * scale, at.y), vec3(.09, .19, .08) * scale, IR_LEAVES, hit);
}

void roomPendant(RoomRay ray, vec3 at, float ceiling, inout RoomHit hit) {
    roomCylinder(ray, at, .24, .20, IR_SHADE, hit);
    float length = max(ceiling - at.y - .20, .02);
    roomBox(ray, vec3(at.x, ceiling - length * .5, at.z), vec3(.012, length * .5, .012), IR_METAL, hit);
}

void officeFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(5, h);
    float w = s.x, d = s.z;
    float shift = (h.x - .5) * min(w * .15, .7);
    if (floorplan == 0) {
        // Two workstations and a fabric privacy screen, with a clear aisle.
        float width = min(1.65, w * .30);
        roomDesk(ray, vec2(w * .26, d * .40), width, true, hit);
        roomDesk(ray, vec2(w * .73, d * .59), width, true, hit);
        roomBox(ray, vec3(w * .51, .77, d * .60), vec3(.045, .73, min(1.05, d * .21)), IR_FABRIC, hit);
        roomBox(ray, vec3(w * .26, .90, d - .30), vec3(w * .21, .90, .27), IR_BOOKS, hit);
    } else if (floorplan == 1) {
        // Private L-desk, visitor chair and tall shelves.
        roomDesk(ray, vec2(w * .43 + shift, d * .47), min(1.90, w * .46), true, hit);
        roomTable(ray, vec2(w * .20, d * .65), vec2(.34, min(.85, d * .22)), .77, hit);
        roomChair(ray, vec2(w * .66, d * .24), vec2(0.0, -1.0), true, hit);
        roomBox(ray, vec3(w - .60, 1.02, d - .28), vec3(.48, 1.02, .25), IR_BOOKS, hit);
        roomPlant(ray, vec2(.46, d - .49), .91, hit);
    } else if (floorplan == 2) {
        // Meeting room with six chairs, display and a low credenza.
        float x = w * .49 + shift, z = d * .49;
        roomTable(ray, vec2(x, z), vec2(min(.78, w * .21), min(1.42, d * .25)), .76, hit);
        for (int i = 0; i < 3; ++i) {
            float dz = (float(i) - 1.0) * min(.89, d * .16);
            roomChair(ray, vec2(x - min(1.15, w * .33), z + dz), vec2(-1.0, 0.0), true, hit);
            roomChair(ray, vec2(x + min(1.15, w * .33), z + dz), vec2(1.0, 0.0), true, hit);
        }
        roomBox(ray, vec3(w * .52, .44, d - .27), vec3(w * .27, .43, .24), IR_CABINET, hit);
        roomBox(ray, vec3(w * .52, 1.75, d - .10), vec3(min(1.03, w * .30), .56, .04), IR_SCREEN, hit);
    } else if (floorplan == 3) {
        // Shared studio: staggered benches, high storage, open floor at the glass.
        roomDesk(ray, vec2(w * .28, d * .31), min(1.55, w * .31), true, hit);
        roomDesk(ray, vec2(w * .70, d * .70), min(1.55, w * .31), false, hit);
        roomChair(ray, vec2(w * .77, d * .43), vec2(1.0, 0.0), true, hit);
        roomBox(ray, vec3(.28, 1.1, d * .76), vec3(.25, 1.1, min(.95, d * .22)), IR_BOOKS, hit);
        roomBox(ray, vec3(w * .50, .62, d * .85), vec3(w * .20, .60, .28), IR_CABINET, hit);
    } else if (floorplan == 4) {
        // Breakout lounge with kitchenette, sofa and a small round table.
        roomSofa(ray, vec2(w * .38, d * .57), min(2.55, w * .58), hit);
        roomRoundTable(ray, vec2(w * .39, d * .30), .46, hit);
        roomBox(ray, vec3(w * .48, .46, d - .36), vec3(w * .33, .46, .32), IR_CABINET, hit);
        roomBox(ray, vec3(w * .48, 1.85, d - .20), vec3(w * .32, .35, .17), IR_CABINET, hit);
        roomPlant(ray, vec2(w - .55, d * .53), 1.04, hit);
    } else {
        // Library / quiet work area, with a different silhouette from open plan.
        roomBox(ray, vec3(w * .27, 1.08, d - .27), vec3(w * .22, 1.08, .25), IR_BOOKS, hit);
        roomBox(ray, vec3(w * .78, 1.08, d - .27), vec3(w * .16, 1.08, .25), IR_BOOKS, hit);
        roomDesk(ray, vec2(w * .63, d * .38), min(1.70, w * .42), true, hit);
        roomBox(ray, vec3(.32, .62, d * .45), vec3(.29, .61, d * .19), IR_CABINET, hit);
        roomPlant(ray, vec2(w * .23, d * .27), .92, hit);
    }
}

void homeFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(6, h);
    float w = s.x, d = s.z;
    if (floorplan == 0 || floorplan == 5) {
        // Living room; the second version adds a reading area instead of a TV.
        roomSofa(ray, vec2(w * .45, d * .59), min(2.75, w * .65), hit);
        roomTable(ray, vec2(w * .45, d * .35), vec2(min(.77, w * .23), .38), .43, hit);
        if (floorplan == 0) {
            roomBox(ray, vec3(w * .48, .39, d - .29), vec3(w * .30, .38, .27), IR_CABINET, hit);
            roomBox(ray, vec3(w * .49, 1.03, d - .23), vec3(w * .22, .36, .04), IR_SCREEN, hit);
        } else {
            roomBox(ray, vec3(w * .28, 1.08, d - .24), vec3(w * .21, 1.08, .22), IR_BOOKS, hit);
            roomChair(ray, vec2(w * .79, d * .32), vec2(1.0, 0.0), true, hit);
        }
        roomPlant(ray, vec2(w - .48, d * .75), .86, hit);
    } else if (floorplan == 1 || floorplan == 4) {
        // Double bedroom / twin-width guest room. The bed is made from solids.
        float x = w * (floorplan == 1 ? .48 : .36), z = d * .56;
        float hw = min(floorplan == 1 ? .88 : .67, w * .26), hz = min(1.05, d * .32);
        roomBox(ray, vec3(x, .22, z), vec3(hw + .045, .19, hz + .035), IR_WOOD, hit);
        roomSoftBox(ray, vec3(x, .52, z), vec3(hw, .16, hz), .09, IR_LINEN, hit);
        roomBox(ray, vec3(x, .77, z + hz), vec3(hw + .075, .60, .07), IR_FABRIC, hit);
        roomSoftBox(ray, vec3(x - hw * .48, .73, z + hz * .63), vec3(hw * .43, .11, .27), .085, IR_PILLOW, hit);
        roomSoftBox(ray, vec3(x + hw * .48, .73, z + hz * .63), vec3(hw * .43, .11, .27), .085, IR_PILLOW, hit);
        roomBox(ray, vec3(x + hw + .38, .31, z + hz * .63), vec3(.27, .30, .29), IR_CABINET, hit);
        roomCylinder(ray, vec3(x + hw + .38, .95, z + hz * .63), .16, .19, IR_SHADE, hit);
        roomBox(ray, vec3(w - .35, 1.14, d * .34), vec3(.32, 1.14, min(.81, d * .20)), IR_CABINET, hit);
    } else if (floorplan == 2) {
        // Eat-in kitchen, with worktop, wall cabinets and four dining chairs.
        roomBox(ray, vec3(w * .48, .47, d - .37), vec3(w * .36, .46, .34), IR_CABINET, hit);
        roomBox(ray, vec3(w * .48, .96, d - .37), vec3(w * .37, .035, .35), IR_WOOD, hit);
        roomBox(ray, vec3(w * .36, 1.94, d - .21), vec3(w * .24, .36, .18), IR_CABINET, hit);
        roomBox(ray, vec3(w * .84, 1.0, d - .40), vec3(.29, .99, .35), IR_PLASTER, hit);
        vec2 at = vec2(w * .45, d * .39);
        roomTable(ray, at, vec2(min(.68, w * .21), .64), .75, hit);
        roomChair(ray, at + vec2(-.92, .0), vec2(-1.0, 0.0), false, hit);
        roomChair(ray, at + vec2(.92, .0), vec2(1.0, 0.0), false, hit);
        roomChair(ray, at + vec2(0.0, -.95), vec2(0.0, -1.0), false, hit);
        roomChair(ray, at + vec2(0.0, .95), vec2(0.0, 1.0), false, hit);
    } else {
        // Study / studio apartment with a work desk and upholstered daybed.
        roomDesk(ray, vec2(w * .29, d * .38), min(1.5, w * .39), true, hit);
        roomSofa(ray, vec2(w * .56, d * .79), min(2.25, w * .56), hit);
        roomBox(ray, vec3(w - .28, 1.06, d * .47), vec3(.25, 1.06, min(.70, d * .18)), IR_BOOKS, hit);
        roomPlant(ray, vec2(w * .78, d * .20), .76, hit);
    }
}

void shopFurniture(RoomRay ray, vec3 s, int kind, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(kind, h);
    float w = s.x, d = s.z;
    if (kind == 0) {
        // Cafe: counter seating, bistro tables, or the reference's sofa / loft.
        roomBox(ray, vec3(w * .36, .54, d * .80), vec3(w * .28, .53, .43), IR_CABINET, hit);
        roomBox(ray, vec3(w * .36, 1.11, d * .80), vec3(w * .29, .045, .46), IR_WOOD, hit);
        roomBox(ray, vec3(w * .27, 1.36, d * .80), vec3(.36, .21, .25), IR_METAL, hit);
        float shelfY = floorplan == 0 ? 1.58 : 1.83;
        roomBox(ray, vec3(w * .35, shelfY, d - .17),
                vec3(w * .27, floorplan == 0 ? .31 : .48, .14), IR_STOCK, hit);
        if (floorplan == 2) {
            roomRoundTable(ray, vec2(w * .25, d * .31), .51, hit);
            roomRoundTable(ray, vec2(w * .73, d * .49), .47, hit);
            roomChair(ray, vec2(w * .25 - .74, d * .31), vec2(-1.0, 0.0), false, hit);
            roomChair(ray, vec2(w * .25 + .74, d * .31), vec2(1.0, 0.0), false, hit);
            roomChair(ray, vec2(w * .73, d * .49 - .75), vec2(0.0, -1.0), false, hit);
        } else if (floorplan == 1) {
            roomTable(ray, vec2(w * .47, d * .39), vec2(w * .30, .50), .77, hit);
            for (int i = 0; i < 3; ++i) {
                float x = w * (.25 + float(i) * .22);
                roomChair(ray, vec2(x, d * .39 + .84), vec2(0.0, 1.0), true, hit);
            }
            roomBox(ray, vec3(w * .87, .51, d * .62), vec3(.30, .49, d * .23), IR_FABRIC, hit);
        } else {
            roomSofa(ray, vec2(w * .27, d * .44), min(2.85, w * .45), hit);
            roomTable(ray, vec2(w * .27, d * .25), vec2(min(.95, w * .20), .46), .48, hit);
            roomBox(ray, vec3(w * .28, .53, d * .25), vec3(.22, .017, .15), IR_PILLOW, hit);
            roomCylinder(ray, vec3(w * .27 - .26, .58, d * .25), .05, .06, IR_POT, hit);
            roomChair(ray, vec2(w * .80, d * .30), vec2(1.0, 0.0), true, hit);
            // A short back stair and landing, framed by a substantial plaster pier.
            float rise = min(2.10, s.y - 1.10), run = d * .34 / 8.0;
            for (int i = 0; i < 8; ++i) {
                float y = rise * (float(i) + 1.0) / 8.0;
                roomBox(ray, vec3(w * .83, y * .5, d * .50 + (float(i) + .5) * run),
                        vec3(w * .10, y * .5, run * .5), IR_WOOD, hit);
            }
            roomBox(ray, vec3(w * .83, rise - .045, d * .92), vec3(w * .155, .065, d * .08), IR_WOOD, hit);
            roomBox(ray, vec3(w * .36, rise - .045, d * .92), vec3(w * .32, .065, d * .08), IR_WOOD, hit);
            roomBox(ray, vec3(w * .37, rise + .62, d * .845), vec3(w * .31, .03, .028), IR_WOOD, hit);
            for (int i = 0; i < 7; ++i)
                roomBox(ray, vec3(w * (.08 + float(i) * .09), rise + .32, d * .845),
                        vec3(.018, .29, .018), IR_WOOD, hit);
            roomBox(ray, vec3(w * .68, rise + .38, d * .90), vec3(.035, .38, .035), IR_WOOD, hit);
            roomBox(ray, vec3(w * .68, rise + .75, d * .92), vec3(.045, .035, d * .08), IR_WOOD, hit);
            roomBox(ray, vec3(w * .65, s.y * .50, d * .65), vec3(.18, s.y * .50, .25), IR_PLASTER, hit);
        }
        for (int i = 0; i < 3; ++i)
            roomBox(ray, vec3(w * .50, s.y - .065, d * (.24 + float(i) * .28)),
                    vec3(w * .50, .065, .095), IR_PLASTER, hit);
        roomPlant(ray, vec2(w - .48, d * .84), 1.0, hit);
    } else if (kind == 1) {
        roomBox(ray, vec3(w * .36, 1.02, d - .25), vec3(w * .29, 1.02, .22),
                floorplan == 1 ? IR_BOOKS : IR_STOCK, hit);
        roomBox(ray, vec3(.26, .94, d * .55), vec3(.23, .93, d * .25), IR_STOCK, hit);
        if (floorplan == 0) {
            roomTable(ray, vec2(w * .52, d * .38), vec2(w * .23, .64), .85, hit);
            roomBox(ray, vec3(w * .43, 1.05, d * .38), vec3(.33, .16, .25), IR_LINEN, hit);
            roomBox(ray, vec3(w * .62, .99, d * .40), vec3(.25, .10, .28), IR_FABRIC, hit);
        } else if (floorplan == 1) {
            roomBox(ray, vec3(w * .49, .74, d * .43), vec3(.40, .73, d * .20), IR_BOOKS, hit);
            roomBox(ray, vec3(w * .81, .55, d * .75), vec3(.49, .54, .41), IR_CABINET, hit);
        } else {
            roomBox(ray, vec3(w * .39, .47, d * .35), vec3(.49, .46, .44), IR_PLASTER, hit);
            roomBox(ray, vec3(w * .73, .67, d * .54), vec3(.42, .66, .39), IR_PLASTER, hit);
            roomCylinder(ray, vec3(w * .39, 1.18, d * .35), .22, .25, IR_POT, hit);
            roomPlant(ray, vec2(w * .73, d * .54), 1.05, hit);
        }
    } else if (kind == 2) {
        roomBox(ray, vec3(w * .46, 1.05, d - .29), vec3(w * .39, 1.04, .26), IR_STOCK, hit);
        float offset = floorplan == 1 ? .10 : .0;
        roomBox(ray, vec3(w * (.28 + offset), .73, d * .52), vec3(.44, .72, d * .25), IR_STOCK, hit);
        if (floorplan != 2)
            roomBox(ray, vec3(w * .72, .73, d * .67), vec3(.44, .72, d * .21), IR_STOCK, hit);
        else {
            roomBox(ray, vec3(w * .68, .47, d * .49), vec3(w * .19, .46, .67), IR_WOOD, hit);
            roomBox(ray, vec3(w * .68, 1.01, d * .49), vec3(w * .18, .08, .63), IR_STOCK, hit);
        }
        roomBox(ray, vec3(w * .80, .51, d * .19), vec3(w * .13, .50, .43), IR_CABINET, hit);
        roomBox(ray, vec3(w * .80, 1.20, d * .22), vec3(.23, .17, .055), IR_SCREEN, hit);
    } else if (kind == 3) {
        // Concourse: substantial side columns, benches and planters.
        for (int i = 0; i < 3; ++i) {
            float z = d * (.26 + float(i) * .25);
            roomBox(ray, vec3(.30, s.y * .5, z), vec3(.21, s.y * .5, .23), IR_PLASTER, hit);
            roomBox(ray, vec3(w - .30, s.y * .5, z), vec3(.21, s.y * .5, .23), IR_PLASTER, hit);
        }
        roomSofa(ray, vec2(w * .28, d * .50), min(1.90, w * .35), hit);
        roomPlant(ray, vec2(w * .78, d * .36), 1.18, hit);
        if (floorplan == 1) roomBox(ray, vec3(w * .69, .56, d * .73), vec3(.55, .55, .65), IR_CABINET, hit);
        if (floorplan == 2) {
            roomSofa(ray, vec2(w * .72, d * .73), min(1.90, w * .35), hit);
            roomPlant(ray, vec2(w * .26, d * .23), 1.05, hit);
        }
    } else {
        // Lobby: reception, waiting seats and a contrasting circulation route.
        float x = w * (floorplan == 1 ? .67 : .35);
        roomBox(ray, vec3(x, .57, d * .68), vec3(w * .22, .56, .45), IR_CABINET, hit);
        roomBox(ray, vec3(x, 1.16, d * .68), vec3(w * .23, .035, .47), IR_WOOD, hit);
        roomBox(ray, vec3(x, 1.38, d * .73), vec3(.29, .19, .035), IR_SCREEN, hit);
        if (floorplan == 2) {
            roomSofa(ray, vec2(w * .55, d * .34), min(2.60, w * .46), hit);
            roomRoundTable(ray, vec2(w * .55, d * .19), .43, hit);
        } else {
            roomChair(ray, vec2(w * .77, d * .32), vec2(1.0, 0.0), true, hit);
            roomChair(ray, vec2(w * .77, d * .47), vec2(1.0, 0.0), true, hit);
        }
        roomPlant(ray, vec2(.52, d * .29), 1.14, hit);
    }
}

void interiorFurniture(RoomRay ray, vec3 size, int kind, vec4 h, inout RoomHit hit) {
    // Very narrow residual suites stay usable, with a cabinet rather than clipped
    // full-size furniture. Normal suites have a bounded, deterministic layout.
    if (size.x < (kind < 5 ? 3.6 : 2.7) || size.z < 3.1) {
        roomBox(ray, vec3(size.x * .5, .55, size.z - .25),
                vec3(size.x * .33, .54, .22), IR_CABINET, hit);
    } else if (kind == 5) officeFurniture(ray, size, h, hit);
    else if (kind == 6) homeFurniture(ray, size, h, hit);
    else shopFurniture(ray, size, kind, h, hit);
    if (kind == 0 || kind == 6) {
        vec3 a, b; interiorLamps(size, kind, h, a, b);
        roomPendant(ray, a, size.y, hit);
        roomPendant(ray, b, size.y, hit);
    }
}
"""
