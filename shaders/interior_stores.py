"""Seeded layouts for shopping-mall units, grocery / convenience stores, restaurants,
pharmacies, gyms and hotel rooms, plus the kind dispatcher.

Concatenated after interior_layouts.INTERIOR_LAYOUTS: it reuses roomChair, roomTable,
roomSofa, roomStool, roomPlant ... and the ray-hit primitives from interiors.py.
Room axes: X along the facade, Y up, Z into the building (the glass is at Z = 0).
"""

INTERIOR_STORES = """
// ---------------------------------------------------------------- shared props
void roomGondola(RoomRay ray, vec2 at, vec2 halfSize, float height, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, height * .5, at.y), vec3(halfSize.x, height * .5, halfSize.y), IR_STOCK, hit);
    roomBox(ray, vec3(at.x, .06, at.y), vec3(halfSize.x + .018, .06, halfSize.y + .018), IR_METAL, hit);
    roomBox(ray, vec3(at.x, height + .014, at.y), vec3(halfSize.x + .014, .014, halfSize.y + .014), IR_METAL, hit);
}

void roomFridgeRun(RoomRay ray, vec2 at, vec2 halfSize, float height, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, height * .5, at.y), vec3(halfSize.x, height * .5, halfSize.y), IR_FRIDGE, hit);
    roomBox(ray, vec3(at.x, .05, at.y), vec3(halfSize.x + .012, .05, halfSize.y + .012), IR_METAL, hit);
}

void roomCounter(RoomRay ray, vec2 at, vec2 halfSize, bool till, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .46, at.y), vec3(halfSize.x, .46, halfSize.y), IR_CABINET, hit);
    roomBox(ray, vec3(at.x, .935, at.y), vec3(halfSize.x + .03, .03, halfSize.y + .03), IR_COUNTER, hit);
    if (till) {
        roomBox(ray, vec3(at.x, 1.14, at.y + halfSize.y * .30), vec3(.17, .14, .022), IR_METAL, hit);
        roomBox(ray, vec3(at.x + halfSize.x * .30, .985, at.y - halfSize.y * .1), vec3(.14, .02, .12), IR_METAL, hit);
    }
}

void roomHangSign(RoomRay ray, vec3 at, vec2 halfSize, float ceiling, inout RoomHit hit) {
    roomBox(ray, at, vec3(halfSize.x, halfSize.y, .03), IR_SIGN, hit);
    float rod = max(ceiling - at.y - halfSize.y, .02);
    roomBox(ray, vec3(at.x, ceiling - rod * .5, at.z), vec3(.008, rod * .5, .008), IR_METAL, hit);
}

void roomClothesRail(RoomRay ray, vec2 at, float halfLen, inout RoomHit hit) {
    roomBox(ray, vec3(at.x - halfLen, .77, at.y), vec3(.012, .77, .012), IR_METAL, hit);
    roomBox(ray, vec3(at.x + halfLen, .77, at.y), vec3(.012, .77, .012), IR_METAL, hit);
    roomBox(ray, vec3(at.x, 1.52, at.y), vec3(halfLen, .012, .012), IR_METAL, hit);
    roomBox(ray, vec3(at.x, 1.13, at.y), vec3(halfLen - .03, .38, .13), IR_CLOTHES, hit);
    roomBox(ray, vec3(at.x - halfLen, .015, at.y), vec3(.03, .015, .27), IR_METAL, hit);
    roomBox(ray, vec3(at.x + halfLen, .015, at.y), vec3(.03, .015, .27), IR_METAL, hit);
}

void roomMannequin(RoomRay ray, vec2 at, inout RoomHit hit) {
    roomCylinder(ray, vec3(at.x, .012, at.y), .21, .012, IR_METAL, hit);
    roomCylinder(ray, vec3(at.x, .60, at.y), .018, .60, IR_METAL, hit);
    roomCylinder(ray, vec3(at.x, .80, at.y), .17, .30, IR_CLOTHES, hit);
    roomEllipsoid(ray, vec3(at.x, 1.28, at.y), vec3(.17, .26, .10), IR_CLOTHES, hit);
    roomEllipsoid(ray, vec3(at.x, 1.66, at.y), vec3(.075, .105, .085), IR_PILLOW, hit);
}

void roomDisplayTable(RoomRay ray, vec2 at, vec2 halfSize, inout RoomHit hit) {
    roomTable(ray, at, halfSize, .80, hit);
    roomBox(ray, vec3(at.x, .875, at.y), vec3(halfSize.x * .84, .04, halfSize.y * .78), IR_CLOTHES, hit);
}

void roomBooth(RoomRay ray, float wallX, float dir, float z, float halfLen, inout RoomHit hit) {
    roomSoftBox(ray, vec3(wallX + dir * .34, .25, z), vec3(.30, .19, halfLen), .06, IR_FABRIC, hit);
    roomSoftBox(ray, vec3(wallX + dir * .12, .72, z), vec3(.09, .40, halfLen), .05, IR_FABRIC, hit);
}

void roomBed(RoomRay ray, vec2 at, float hw, float hz, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .22, at.y), vec3(hw + .045, .19, hz + .035), IR_WOOD, hit);
    roomSoftBox(ray, vec3(at.x, .52, at.y), vec3(hw, .16, hz), .09, IR_LINEN, hit);
    roomSoftBox(ray, vec3(at.x, .595, at.y - hz * .42), vec3(hw + .012, .075, hz * .52), .05, IR_FABRIC, hit);
    roomBox(ray, vec3(at.x, .80, at.y + hz), vec3(hw + .09, .62, .06), IR_FABRIC, hit);
    roomSoftBox(ray, vec3(at.x - hw * .48, .73, at.y + hz * .66), vec3(hw * .42, .11, .26), .085, IR_PILLOW, hit);
    roomSoftBox(ray, vec3(at.x + hw * .48, .73, at.y + hz * .66), vec3(hw * .42, .11, .26), .085, IR_PILLOW, hit);
}

void roomBedside(RoomRay ray, vec2 at, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .27, at.y), vec3(.26, .26, .24), IR_CABINET, hit);
    roomCylinder(ray, vec3(at.x, .74, at.y), .15, .17, IR_SHADE, hit);
    roomCylinder(ray, vec3(at.x, .56, at.y), .02, .04, IR_METAL, hit);
}

void roomTreadmill(RoomRay ray, vec2 at, bool facingGlass, inout RoomHit hit) {
    float f = facingGlass ? -1.0 : 1.0;
    roomBox(ray, vec3(at.x, .15, at.y), vec3(.37, .09, .88), IR_RUBBER, hit);
    roomBox(ray, vec3(at.x, .235, at.y), vec3(.29, .012, .78), IR_METAL, hit);
    roomBox(ray, vec3(at.x - .35, .70, at.y + f * .80), vec3(.022, .55, .03), IR_METAL, hit);
    roomBox(ray, vec3(at.x + .35, .70, at.y + f * .80), vec3(.022, .55, .03), IR_METAL, hit);
    roomBox(ray, vec3(at.x, 1.25, at.y + f * .80), vec3(.29, .11, .04), facingGlass ? IR_METAL : IR_SCREEN, hit);
}

void roomWeightBench(RoomRay ray, vec2 at, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .44, at.y), vec3(.22, .055, .56), IR_RUBBER, hit);
    roomBox(ray, vec3(at.x - .15, .21, at.y), vec3(.025, .21, .40), IR_METAL, hit);
    roomBox(ray, vec3(at.x + .15, .21, at.y), vec3(.025, .21, .40), IR_METAL, hit);
    roomBox(ray, vec3(at.x - .55, .65, at.y + .62), vec3(.025, .65, .025), IR_METAL, hit);
    roomBox(ray, vec3(at.x + .55, .65, at.y + .62), vec3(.025, .65, .025), IR_METAL, hit);
    roomBox(ray, vec3(at.x, 1.28, at.y + .62), vec3(.95, .014, .014), IR_METAL, hit);
    roomBox(ray, vec3(at.x - .74, 1.28, at.y + .62), vec3(.03, .22, .22), IR_RUBBER, hit);
    roomBox(ray, vec3(at.x + .74, 1.28, at.y + .62), vec3(.03, .22, .22), IR_RUBBER, hit);
}

void roomDumbbellRack(RoomRay ray, vec2 at, float halfLen, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .20, at.y), vec3(halfLen, .20, .20), IR_METAL, hit);
    roomBox(ray, vec3(at.x, .55, at.y - .02), vec3(halfLen - .03, .07, .11), IR_RUBBER, hit);
    roomBox(ray, vec3(at.x, .78, at.y - .02), vec3(halfLen - .03, .07, .10), IR_RUBBER, hit);
}

void roomMachine(RoomRay ray, vec2 at, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .04, at.y), vec3(.45, .04, .70), IR_METAL, hit);
    roomSoftBox(ray, vec3(at.x, .47, at.y - .10), vec3(.23, .07, .23), .04, IR_FABRIC, hit);
    roomSoftBox(ray, vec3(at.x, .80, at.y + .12), vec3(.22, .26, .06), .04, IR_FABRIC, hit);
    roomBox(ray, vec3(at.x, .90, at.y + .52), vec3(.20, .86, .09), IR_RUBBER, hit);
    roomBox(ray, vec3(at.x - .26, .95, at.y + .52), vec3(.025, .95, .06), IR_METAL, hit);
    roomBox(ray, vec3(at.x + .26, .95, at.y + .52), vec3(.025, .95, .06), IR_METAL, hit);
    roomBox(ray, vec3(at.x, 1.92, at.y + .52), vec3(.28, .03, .07), IR_METAL, hit);
}

void roomBike(RoomRay ray, vec2 at, inout RoomHit hit) {
    roomBox(ray, vec3(at.x, .03, at.y), vec3(.28, .03, .52), IR_METAL, hit);
    roomBox(ray, vec3(at.x, .40, at.y + .24), vec3(.075, .27, .22), IR_METAL, hit);
    roomBox(ray, vec3(at.x, .80, at.y - .12), vec3(.025, .30, .025), IR_METAL, hit);
    roomSoftBox(ray, vec3(at.x, 1.12, at.y - .16), vec3(.075, .035, .13), .02, IR_RUBBER, hit);
    roomBox(ray, vec3(at.x, 1.02, at.y + .27), vec3(.025, .22, .025), IR_METAL, hit);
    roomBox(ray, vec3(at.x, 1.25, at.y + .27), vec3(.20, .018, .018), IR_METAL, hit);
}

// ---------------------------------------------------------------- shopping mall unit
void mallFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(7, h);
    float w = s.x, d = s.z;
    float top = min(2.30, s.y - .45);
    if (floorplan == 0) {
        // Fashion: wall display, rails, mannequins at the glass, folded-stock table, till.
        roomBox(ray, vec3(w * .50, .42, d - .22), vec3(w * .45, .42, .20), IR_CABINET, hit);
        roomBox(ray, vec3(w * .50, (top + .84) * .5, d - .16), vec3(w * .45, (top - .84) * .5, .14), IR_CLOTHES, hit);
        roomClothesRail(ray, vec2(w * .27, d * .66), min(.95, w * .16), hit);
        roomClothesRail(ray, vec2(w * .66, d * .72), min(.95, w * .16), hit);
        roomMannequin(ray, vec2(w * .26, 1.00), hit);
        roomMannequin(ray, vec2(w * .40, 1.30), hit);
        roomDisplayTable(ray, vec2(w * .70, d * .36), vec2(min(.85, w * .15), .52), hit);
        roomCounter(ray, vec2(w - .55, d * .50), vec2(.32, .75), true, hit);
    } else if (floorplan == 1) {
        // Electronics: screen wall, demo islands, service counter.
        float sw = min(.58, w * .11);
        for (int i = 0; i < 3; ++i)
            roomBox(ray, vec3(w * (.24 + float(i) * .26), 1.62, d - .06), vec3(sw, .34, .03), IR_SCREEN, hit);
        roomBox(ray, vec3(w * .50, .42, d - .22), vec3(w * .45, .42, .20), IR_CABINET, hit);
        for (int i = 0; i < 2; ++i) {
            vec2 at = i == 0 ? vec2(w * .30, d * .40) : vec2(w * .70, d * .58);
            roomBox(ray, vec3(at.x, .50, at.y), vec3(.68, .50, .44), IR_COUNTER, hit);
            roomBox(ray, vec3(at.x - .28, 1.16, at.y), vec3(.22, .15, .016), IR_SCREEN, hit);
            roomBox(ray, vec3(at.x + .26, 1.16, at.y), vec3(.22, .15, .016), IR_SCREEN, hit);
        }
        roomCounter(ray, vec2(w - .50, d * .22), vec2(.34, .70), true, hit);
    } else if (floorplan == 2) {
        // Sports / footwear: perimeter walls, bench, shoe tower, one mannequin.
        roomBox(ray, vec3(.20, top * .5, d * .52), vec3(.18, top * .5, d * .36), IR_STOCK, hit);
        roomBox(ray, vec3(w - .20, top * .5, d * .52), vec3(.18, top * .5, d * .36), IR_STOCK, hit);
        roomBox(ray, vec3(w * .50, top * .5, d - .18), vec3(w * .38, top * .5, .16), IR_STOCK, hit);
        roomSoftBox(ray, vec3(w * .50, .22, d * .38), vec3(.95, .20, .28), .05, IR_FABRIC, hit);
        roomBox(ray, vec3(w * .50, .62, d * .66), vec3(.32, .62, .26), IR_STOCK, hit);
        roomMannequin(ray, vec2(w * .80, 1.10), hit);
    } else if (floorplan == 3) {
        // Cosmetics / jewellery: lit glass cases at the glass, lit back wall, till.
        roomBox(ray, vec3(w * .50, top * .5, d - .15), vec3(w * .42, top * .5, .13), IR_STOCK, hit);
        roomFridgeRun(ray, vec2(w * .30, d * .30), vec2(.56, .36), 1.0, hit);
        roomFridgeRun(ray, vec2(w * .70, d * .48), vec2(.56, .36), 1.0, hit);
        roomCounter(ray, vec2(w * .50, d * .76), vec2(min(1.3, w * .26), .36), true, hit);
        roomStool(ray, vec2(w * .50 - .55, d * .76 + .62), .72, hit);
    } else if (floorplan == 4) {
        // Bakery / food kiosk: serving counter with lit display, menu board, window stools.
        roomBox(ray, vec3(w * .50, .50, d * .60), vec3(w * .38, .50, .36), IR_CABINET, hit);
        roomBox(ray, vec3(w * .50, 1.03, d * .60), vec3(w * .39, .03, .38), IR_COUNTER, hit);
        roomFridgeRun(ray, vec2(w * .50, d * .60 - .02), vec2(w * .33, .28), 1.55, hit);
        roomBox(ray, vec3(w * .50, 1.95, d - .05), vec3(w * .30, .45, .03), IR_SCREEN, hit);
        roomBox(ray, vec3(w * .50, 1.0, d - .20), vec3(w * .42, 1.0, .17), IR_STOCK, hit);
        roomTable(ray, vec2(w * .80, d * .22), vec2(.45, .38), 1.0, hit);
        roomStool(ray, vec2(w * .80 - .55, d * .22), .72, hit);
        roomStool(ray, vec2(w * .80 + .55, d * .22), .72, hit);
    } else {
        // Homeware: sofa vignette, floor lamp, bookshelf wall, dining set.
        roomBox(ray, vec3(w * .50, top * .5, d - .17), vec3(w * .42, top * .5, .15), IR_BOOKS, hit);
        roomSofa(ray, vec2(w * .30, d * .46), min(2.40, w * .42), hit);
        roomTable(ray, vec2(w * .30, d * .28), vec2(min(.70, w * .12), .36), .42, hit);
        roomBox(ray, vec3(w * .60, .70, d * .22), vec3(.012, .70, .012), IR_METAL, hit);
        roomCylinder(ray, vec3(w * .60, 1.50, d * .22), .20, .17, IR_SHADE, hit);
        roomTable(ray, vec2(w * .76, d * .60), vec2(.56, .40), .75, hit);
        roomChair(ray, vec2(w * .76 - .78, d * .60), vec2(-1.0, 0.0), false, hit);
        roomChair(ray, vec2(w * .76 + .78, d * .60), vec2(1.0, 0.0), false, hit);
        roomPlant(ray, vec2(w - .45, d * .88), 1.0, hit);
    }
}

// ---------------------------------------------------------------- grocery / convenience
void groceryFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(8, h);
    float w = s.x, d = s.z;
    float signY = min(s.y - .42, 2.30);
    if (floorplan == 0) {
        // Three aisles running into the store, fridge wall behind, till by the door.
        for (int i = 0; i < 3; ++i) {
            float x = w * (.20 + float(i) * .30);
            roomGondola(ray, vec2(x, d * .53), vec2(.30, d * .21), 1.55, hit);
            roomHangSign(ray, vec3(x, signY, d * .31), vec2(.34, .17), s.y, hit);
        }
        roomFridgeRun(ray, vec2(w * .50, d - .32), vec2(w * .44, .30), 2.0, hit);
        roomCounter(ray, vec2(w * .84, .95), vec2(.58, .34), true, hit);
    } else if (floorplan == 1) {
        // Cross aisles seen long-side from the glass, fridge wall on the side.
        roomGondola(ray, vec2(w * .50, d * .30), vec2(w * .28, .30), 1.50, hit);
        roomGondola(ray, vec2(w * .54, d * .53), vec2(w * .28, .30), 1.50, hit);
        roomGondola(ray, vec2(w * .50, d * .76), vec2(w * .28, .30), 1.50, hit);
        roomFridgeRun(ray, vec2(.31, d * .50), vec2(.29, d * .36), 2.0, hit);
        roomHangSign(ray, vec3(w * .50, signY, d * .30 - .36), vec2(.40, .17), s.y, hit);
        roomHangSign(ray, vec3(w * .54, signY, d * .53 - .36), vec2(.40, .17), s.y, hit);
    } else if (floorplan == 2) {
        // Convenience store: till and tobacco wall near the door, two short aisles.
        roomBox(ray, vec3(.18, 1.05, d * .30), vec3(.16, 1.05, d * .22), IR_STOCK, hit);
        roomCounter(ray, vec2(w * .26, .90), vec2(.80, .34), true, hit);
        roomFridgeRun(ray, vec2(w * .56, d - .32), vec2(w * .38, .30), 2.0, hit);
        roomGondola(ray, vec2(w * .54, d * .45), vec2(.80, .28), 1.25, hit);
        roomGondola(ray, vec2(w * .66, d * .66), vec2(.80, .28), 1.25, hit);
        roomBox(ray, vec3(w * .86, .45, d * .22), vec3(.34, .45, .30), IR_STOCK, hit);
    } else if (floorplan == 3) {
        // Fridge-heavy: walls of cold drinks either side, chest freezer island.
        roomFridgeRun(ray, vec2(.31, d * .50), vec2(.29, d * .38), 2.0, hit);
        roomFridgeRun(ray, vec2(w - .31, d * .50), vec2(.29, d * .38), 2.0, hit);
        roomFridgeRun(ray, vec2(w * .50, d - .32), vec2(w * .36, .30), 2.0, hit);
        roomFridgeRun(ray, vec2(w * .50, d * .50), vec2(min(1.0, w * .18), .42), .92, hit);
        roomCounter(ray, vec2(w * .50, .92), vec2(.70, .32), true, hit);
        roomHangSign(ray, vec3(w * .50, signY, d * .50), vec2(.45, .17), s.y, hit);
    } else if (floorplan == 4) {
        // Produce hall: crates of fruit and veg, shelving and chiller behind.
        float cx = min(.62, w * .12);
        for (int i = 0; i < 4; ++i) {
            vec2 at = vec2(w * (i < 2 ? .28 : .70), d * (mod(float(i), 2.0) < .5 ? .36 : .62));
            roomBox(ray, vec3(at.x, .25, at.y), vec3(cx, .25, .46), IR_WOOD, hit);
            roomBox(ray, vec3(at.x, .62, at.y - .04), vec3(cx - .03, .13, .40), IR_STOCK, hit);
        }
        roomFridgeRun(ray, vec2(w * .50, d - .32), vec2(w * .40, .30), 1.9, hit);
        roomHangSign(ray, vec3(w * .50, signY, d * .62), vec2(.50, .17), s.y, hit);
    } else {
        // Promo pallet at the glass, two aisles, fridge wall, till at the back corner.
        roomBox(ray, vec3(w * .50, .55, .80), vec3(min(.85, w * .17), .55, .45), IR_STOCK, hit);
        roomBox(ray, vec3(w * .50, 1.19, .80), vec3(min(.85, w * .17) + .01, .035, .46), IR_SIGN, hit);
        roomGondola(ray, vec2(w * .46, d * .46), vec2(w * .26, .30), 1.50, hit);
        roomGondola(ray, vec2(w * .54, d * .70), vec2(w * .26, .30), 1.50, hit);
        roomFridgeRun(ray, vec2(.31, d * .55), vec2(.29, d * .32), 2.0, hit);
        roomCounter(ray, vec2(w - .55, d * .55), vec2(.34, .80), true, hit);
    }
}

// ---------------------------------------------------------------- restaurant / bar
void restaurantFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(9, h);
    float w = s.x, d = s.z;
    if (floorplan == 0) {
        // Bistro: round tables, wooden chairs, plant.
        vec2 t0 = vec2(w * .27, d * .32), t1 = vec2(w * .70, d * .62), t2 = vec2(w * .30, d * .74);
        roomRoundTable(ray, t0, .44, hit);
        roomRoundTable(ray, t1, .44, hit);
        roomRoundTable(ray, t2, .40, hit);
        roomChair(ray, t0 + vec2(-.72, 0.0), vec2(-1.0, 0.0), false, hit);
        roomChair(ray, t0 + vec2(.72, 0.0), vec2(1.0, 0.0), false, hit);
        roomChair(ray, t1 + vec2(-.72, 0.0), vec2(-1.0, 0.0), false, hit);
        roomChair(ray, t1 + vec2(0.0, -.72), vec2(0.0, -1.0), false, hit);
        roomChair(ray, t2 + vec2(.70, 0.0), vec2(1.0, 0.0), false, hit);
        roomBox(ray, vec3(w * .72, .90, d - .22), vec3(w * .20, .90, .20), IR_STOCK, hit);
        roomPlant(ray, vec2(w - .45, d * .26), 1.05, hit);
    } else if (floorplan == 1) {
        // Booths along a wall, two-tops opposite.
        roomBooth(ray, 0.0, 1.0, d * .30, min(.62, d * .09), hit);
        roomBooth(ray, 0.0, 1.0, d * .62, min(.62, d * .09), hit);
        roomTable(ray, vec2(1.18, d * .30), vec2(.40, min(.62, d * .09)), .75, hit);
        roomTable(ray, vec2(1.18, d * .62), vec2(.40, min(.62, d * .09)), .75, hit);
        roomChair(ray, vec2(1.18 + .70, d * .30), vec2(1.0, 0.0), false, hit);
        roomChair(ray, vec2(1.18 + .70, d * .62), vec2(1.0, 0.0), false, hit);
        roomTable(ray, vec2(w * .74, d * .40), vec2(.36, .36), .75, hit);
        roomChair(ray, vec2(w * .74, d * .40 - .62), vec2(0.0, -1.0), false, hit);
        roomChair(ray, vec2(w * .74, d * .40 + .62), vec2(0.0, 1.0), false, hit);
        roomBox(ray, vec3(w * .60, 1.0, d - .20), vec3(w * .24, 1.0, .18), IR_STOCK, hit);
    } else if (floorplan == 2) {
        // Bar: counter with stools, back-bar bottles, one table at the glass.
        roomBox(ray, vec3(w * .50, .55, d * .66), vec3(w * .36, .55, .32), IR_CABINET, hit);
        roomBox(ray, vec3(w * .50, 1.12, d * .66), vec3(w * .37, .035, .36), IR_WOOD, hit);
        for (int i = 0; i < 4; ++i)
            roomStool(ray, vec2(w * (.28 + float(i) * .147), d * .66 - .66), .74, hit);
        roomBox(ray, vec3(w * .50, 1.35, d - .16), vec3(w * .40, .62, .14), IR_STOCK, hit);
        roomBox(ray, vec3(w * .50, .45, d - .24), vec3(w * .40, .45, .22), IR_CABINET, hit);
        roomRoundTable(ray, vec2(w * .22, d * .26), .42, hit);
        roomChair(ray, vec2(w * .22 + .70, d * .26), vec2(1.0, 0.0), false, hit);
    } else if (floorplan == 3) {
        // Communal long table with benches of chairs.
        float hx = min(w * .30, 1.55);
        roomTable(ray, vec2(w * .50, d * .48), vec2(hx, .46), .76, hit);
        for (int i = 0; i < 3; ++i) {
            float x = w * .50 + (float(i) - 1.0) * hx * .62;
            roomChair(ray, vec2(x, d * .48 - .78), vec2(0.0, -1.0), false, hit);
            roomChair(ray, vec2(x, d * .48 + .78), vec2(0.0, 1.0), false, hit);
        }
        roomBox(ray, vec3(w * .50, 1.1, d - .20), vec3(w * .32, 1.1, .18), IR_STOCK, hit);
        roomPlant(ray, vec2(.45, d * .84), 1.1, hit);
    } else if (floorplan == 4) {
        // Window counter with stools facing the street, tables behind.
        float cx = min(w * .38, 2.2);
        roomBox(ray, vec3(w * .50, 1.06, .52), vec3(cx, .028, .24), IR_WOOD, hit);
        roomBox(ray, vec3(w * .50, .52, .60), vec3(cx - .05, .50, .06), IR_CABINET, hit);
        for (int i = 0; i < 4; ++i)
            roomStool(ray, vec2(w * .50 + (float(i) - 1.5) * cx * .48, .98), .74, hit);
        roomTable(ray, vec2(w * .30, d * .60), vec2(.42, .42), .75, hit);
        roomTable(ray, vec2(w * .72, d * .62), vec2(.42, .42), .75, hit);
        roomChair(ray, vec2(w * .30, d * .60 + .70), vec2(0.0, 1.0), false, hit);
        roomChair(ray, vec2(w * .72, d * .62 + .70), vec2(0.0, 1.0), false, hit);
    } else {
        // Dining room: two four-tops, a service station and plants.
        vec2 a = vec2(w * .27, d * .34), b = vec2(w * .70, d * .62);
        roomTable(ray, a, vec2(.50, .50), .75, hit);
        roomTable(ray, b, vec2(.50, .50), .75, hit);
        for (int i = 0; i < 2; ++i) {
            float sg = float(i) * 2.0 - 1.0;
            roomChair(ray, a + vec2(sg * .85, 0.0), vec2(sg, 0.0), false, hit);
            roomChair(ray, b + vec2(sg * .85, 0.0), vec2(sg, 0.0), false, hit);
            roomChair(ray, a + vec2(0.0, sg * .85), vec2(0.0, sg), false, hit);
            roomChair(ray, b + vec2(0.0, sg * .85), vec2(0.0, sg), false, hit);
        }
        roomBox(ray, vec3(w * .30, .50, d - .24), vec3(w * .20, .50, .22), IR_CABINET, hit);
        roomPlant(ray, vec2(w - .45, d * .22), 1.1, hit);
    }
}

// ---------------------------------------------------------------- pharmacy / health shop
void pharmacyFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(10, h);
    float w = s.x, d = s.z;
    float top = min(2.25, s.y - .40);
    if (floorplan == 0) {
        roomBox(ray, vec3(.19, top * .5, d * .50), vec3(.17, top * .5, d * .36), IR_STOCK, hit);
        roomBox(ray, vec3(w - .19, top * .5, d * .50), vec3(.17, top * .5, d * .36), IR_STOCK, hit);
        roomGondola(ray, vec2(w * .40, d * .42), vec2(.30, d * .15), 1.30, hit);
        roomGondola(ray, vec2(w * .62, d * .42), vec2(.30, d * .15), 1.30, hit);
        roomCounter(ray, vec2(w * .50, d - .85), vec2(w * .28, .34), true, hit);
        roomBox(ray, vec3(w * .50, top * .5, d - .15), vec3(w * .30, top * .5, .13), IR_STOCK, hit);
    } else if (floorplan == 1) {
        roomCounter(ray, vec2(w * .50, d * .60), vec2(w * .36, .36), true, hit);
        roomBox(ray, vec3(w * .50, top * .5, d - .16), vec3(w * .38, top * .5, .14), IR_CABINET, hit);
        roomBox(ray, vec3(w * .50, 1.5, d - .31), vec3(w * .34, .55, .04), IR_STOCK, hit);
        for (int i = 0; i < 3; ++i)
            roomChair(ray, vec2(w * (.28 + float(i) * .22), 1.0), vec2(0.0, -1.0), true, hit);
        roomPlant(ray, vec2(w - .45, d * .30), .95, hit);
    } else if (floorplan == 2) {
        roomGondola(ray, vec2(w * .40, d * .32), vec2(w * .24, .30), 1.60, hit);
        roomGondola(ray, vec2(w * .60, d * .55), vec2(w * .24, .30), 1.60, hit);
        roomFridgeRun(ray, vec2(w * .82, d * .30), vec2(.50, .34), 1.0, hit);
        roomBox(ray, vec3(w * .50, top * .5, d - .16), vec3(w * .42, top * .5, .14), IR_STOCK, hit);
        roomCounter(ray, vec2(w * .20, d * .82), vec2(.5, .34), true, hit);
    } else if (floorplan == 3) {
        roomCounter(ray, vec2(w * .26, .95), vec2(.58, .34), true, hit);
        roomGondola(ray, vec2(w * .52, d * .56), vec2(.30, d * .20), 1.55, hit);
        roomGondola(ray, vec2(w * .78, d * .56), vec2(.30, d * .20), 1.55, hit);
        roomFridgeRun(ray, vec2(w * .50, d - .32), vec2(w * .30, .30), 1.9, hit);
        roomBox(ray, vec3(.19, top * .5, d * .56), vec3(.17, top * .5, d * .22), IR_STOCK, hit);
    } else if (floorplan == 4) {
        float r = min(.95, w * .17);
        roomCylinder(ray, vec3(w * .50, .60, d * .44), r, .60, IR_STOCK, hit);
        roomCylinder(ray, vec3(w * .50, 1.22, d * .44), r + .03, .02, IR_METAL, hit);
        roomBox(ray, vec3(w * .50, top * .5, d - .16), vec3(w * .40, top * .5, .14), IR_STOCK, hit);
        roomBox(ray, vec3(.19, top * .5, d * .56), vec3(.17, top * .5, d * .28), IR_STOCK, hit);
        roomCounter(ray, vec2(w - .60, d * .70), vec2(.34, .70), true, hit);
    } else {
        roomDesk(ray, vec2(w * .72, d * .52), min(1.5, w * .34), true, hit);
        for (int i = 0; i < 3; ++i)
            roomChair(ray, vec2(w * .22, d * (.30 + float(i) * .16)), vec2(-1.0, 0.0), true, hit);
        roomBox(ray, vec3(w * .52, top * .5, d - .16), vec3(w * .34, top * .5, .14), IR_STOCK, hit);
        roomBox(ray, vec3(w - .20, 1.05, d * .30), vec3(.17, 1.05, d * .14), IR_CABINET, hit);
        roomPlant(ray, vec2(.45, d * .20), .95, hit);
    }
}

// ---------------------------------------------------------------- gym / studio
void gymFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(11, h);
    float w = s.x, d = s.z;
    if (floorplan == 0) {
        for (int i = 0; i < 3; ++i) roomTreadmill(ray, vec2(w * (.22 + float(i) * .28), d * .30), true, hit);
        roomWeightBench(ray, vec2(w * .32, d * .74), hit);
        roomDumbbellRack(ray, vec2(w * .74, d - .30), w * .18, hit);
    } else if (floorplan == 1) {
        roomDumbbellRack(ray, vec2(w * .50, d - .30), w * .36, hit);
        roomWeightBench(ray, vec2(w * .28, d * .46), hit);
        roomWeightBench(ray, vec2(w * .70, d * .50), hit);
        roomCylinder(ray, vec3(w - .40, .16, d * .20), .22, .16, IR_RUBBER, hit);
        roomEllipsoid(ray, vec3(.50, .30, d * .22), vec3(.30, .30, .30), IR_FABRIC, hit);
    } else if (floorplan == 2) {
        for (int i = 0; i < 4; ++i)
            roomBox(ray, vec3(w * (.18 + float(i) * .21), .012, d * .46), vec3(.31, .012, .92), IR_FABRIC, hit);
        roomEllipsoid(ray, vec3(w * .12, .32, d * .84), vec3(.32, .32, .32), IR_FABRIC, hit);
        roomEllipsoid(ray, vec3(w * .28, .32, d * .88), vec3(.32, .32, .32), IR_PILLOW, hit);
        roomBox(ray, vec3(w - .30, .55, d * .80), vec3(.26, .55, .50), IR_CABINET, hit);
    } else if (floorplan == 3) {
        roomMachine(ray, vec2(w * .24, d * .40), hit);
        roomMachine(ray, vec2(w * .52, d * .30), hit);
        roomMachine(ray, vec2(w * .78, d * .52), hit);
        roomMachine(ray, vec2(w * .36, d * .70), hit);
    } else if (floorplan == 4) {
        for (int i = 0; i < 3; ++i) {
            roomBike(ray, vec2(w * (.20 + float(i) * .30), d * .36), hit);
            roomBike(ray, vec2(w * (.20 + float(i) * .30), d * .66), hit);
        }
    } else {
        roomBox(ray, vec3(w * .50, .02, d * .50), vec3(w * .42, .02, d * .30), IR_FABRIC, hit);
        for (int i = 0; i < 3; ++i) {
            float x = w * (.24 + float(i) * .26);
            roomBox(ray, vec3(x, s.y - .12, d * .46), vec3(.012, .12, .012), IR_METAL, hit);
            roomCylinder(ray, vec3(x, 1.12, d * .46), .17, .52, IR_RUBBER, hit);
        }
        roomWeightBench(ray, vec2(w * .22, d * .84), hit);
        roomBox(ray, vec3(w - .28, .55, d * .82), vec3(.24, .55, .50), IR_CABINET, hit);
    }
}

// ---------------------------------------------------------------- hotel room
void hotelFurniture(RoomRay ray, vec3 s, vec4 h, inout RoomHit hit) {
    int floorplan = interiorVariant(12, h);
    float w = s.x, d = s.z;
    float hw = min(.90, w * .22), hz = min(1.05, d * .23);
    if (floorplan == 0) {
        roomBed(ray, vec2(w * .46, d * .70), hw, hz, hit);
        roomBedside(ray, vec2(w * .46 - hw - .34, d - .30), hit);
        roomBedside(ray, vec2(w * .46 + hw + .34, d - .30), hit);
        roomBox(ray, vec3(w * .46, 1.25, d - .03), vec3(hw + .45, .55, .025), IR_WOOD, hit);
        roomBox(ray, vec3(.32, 1.14, d * .42), vec3(.30, 1.14, min(.85, d * .17)), IR_CABINET, hit);
        roomDesk(ray, vec2(w - .70, d * .36), 1.2, true, hit);
    } else if (floorplan == 1) {
        float hx = min(.52, w * .12);
        roomBed(ray, vec2(w * .27, d * .72), hx, hz, hit);
        roomBed(ray, vec2(w * .66, d * .72), hx, hz, hit);
        roomBedside(ray, vec2(w * .465, d - .30), hit);
        roomBox(ray, vec3(w - .32, 1.14, d * .36), vec3(.30, 1.14, min(.80, d * .16)), IR_CABINET, hit);
        roomBox(ray, vec3(.30, .55, d * .30), vec3(.26, .55, .45), IR_WOOD, hit);
    } else if (floorplan == 2) {
        roomBed(ray, vec2(w * .50, d * .78), hw, hz * .90, hit);
        roomBedside(ray, vec2(w * .50 - hw - .34, d - .30), hit);
        roomBedside(ray, vec2(w * .50 + hw + .34, d - .30), hit);
        roomSofa(ray, vec2(w * .34, d * .26), min(1.9, w * .40), hit);
        roomTable(ray, vec2(w * .34, d * .12 + .32), vec2(.5, .26), .42, hit);
        roomPlant(ray, vec2(w - .45, d * .30), .95, hit);
    } else if (floorplan == 3) {
        roomBed(ray, vec2(w * .40, d * .72), hw, hz, hit);
        roomBedside(ray, vec2(w * .40 + hw + .34, d - .30), hit);
        roomDesk(ray, vec2(w * .76, d * .24), min(1.3, w * .28), true, hit);
        roomChair(ray, vec2(w * .80, d * .62), vec2(1.0, 0.0), true, hit);
        roomBox(ray, vec3(.30, 1.14, d * .45), vec3(.28, 1.14, min(.85, d * .17)), IR_CABINET, hit);
        roomCylinder(ray, vec3(w * .88, 1.50, d * .62), .20, .17, IR_SHADE, hit);
    } else if (floorplan == 4) {
        roomBed(ray, vec2(w * .66, d * .72), min(.80, w * .18), hz, hit);
        roomBox(ray, vec3(w * .30, .47, d - .37), vec3(w * .27, .46, .34), IR_CABINET, hit);
        roomBox(ray, vec3(w * .30, .96, d - .37), vec3(w * .28, .035, .35), IR_WOOD, hit);
        roomBox(ray, vec3(w * .30, 1.94, d - .21), vec3(w * .24, .36, .18), IR_CABINET, hit);
        roomTable(ray, vec2(w * .26, d * .40), vec2(.36, .36), .75, hit);
        roomChair(ray, vec2(w * .26, d * .40 + .66), vec2(0.0, 1.0), false, hit);
        roomChair(ray, vec2(w * .26 + .66, d * .40), vec2(1.0, 0.0), false, hit);
    } else {
        roomBed(ray, vec2(w * .55, d * .70), hw, hz, hit);
        roomBedside(ray, vec2(w * .55 - hw - .34, d - .30), hit);
        roomChair(ray, vec2(.62, d * .36), vec2(-1.0, 0.0), true, hit);
        roomBox(ray, vec3(.28, .40, d * .56), vec3(.24, .40, .55), IR_CABINET, hit);
        roomBox(ray, vec3(w - .34, 1.14, d * .34), vec3(.32, 1.14, min(.80, d * .17)), IR_CABINET, hit);
    }
}

// ---------------------------------------------------------------- dispatcher
void interiorFurniture(RoomRay ray, vec3 size, int kind, vec4 h, inout RoomHit hit) {
    // Very narrow residual suites stay usable, with a cabinet (or a stocked shelf in a
    // shop) rather than clipped full-size furniture.
    if (size.x < interiorMinWidth(kind) || size.z < interiorMinDepth(kind)) {
        if (interiorIsShop(kind))
            roomBox(ray, vec3(size.x * .5, .85, size.z - .22), vec3(size.x * .38, .85, .20), IR_STOCK, hit);
        else
            roomBox(ray, vec3(size.x * .5, .55, size.z - .25), vec3(size.x * .33, .54, .22), IR_CABINET, hit);
    } else if (kind == IK_OFFICE) officeFurniture(ray, size, h, hit);
    else if (kind == IK_HOME) homeFurniture(ray, size, h, hit);
    else if (kind == IK_MALL) mallFurniture(ray, size, h, hit);
    else if (kind == IK_GROCERY) groceryFurniture(ray, size, h, hit);
    else if (kind == IK_RESTAURANT) restaurantFurniture(ray, size, h, hit);
    else if (kind == IK_PHARMACY) pharmacyFurniture(ray, size, h, hit);
    else if (kind == IK_GYM) gymFurniture(ray, size, h, hit);
    else if (kind == IK_HOTEL) hotelFurniture(ray, size, h, hit);
    else shopFurniture(ray, size, kind, h, hit);
    bool vacant = kind == IK_HOME && interiorVariant(IK_HOME, h) == 6;
    if ((kind == IK_CAFE || kind == IK_HOME || kind == IK_RESTAURANT) && !vacant) {
        vec3 a, b; interiorLamps(size, kind, h, a, b);
        roomPendant(ray, a, size.y, hit);
        roomPendant(ray, b, size.y, hit);
    }
}
"""