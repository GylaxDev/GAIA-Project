"""Finite 3D rooms: ray-hit furniture, filtered materials and practical lighting."""
from .interior_layouts import INTERIOR_LAYOUTS
from .interior_stores import INTERIOR_STORES

# Room kinds understood by interiorRadianceKind(). The ids are shared with the GLSL
# constants IK_* below; the first seven keep their historical numbering.
INTERIOR_KINDS = ('cafe', 'retail', 'market', 'arcade', 'lobby', 'office', 'home',
                  'mall', 'grocery', 'restaurant', 'pharmacy', 'gym', 'hotel')

_SHOP_TO_KIND = {
    'supermarket': 'grocery', 'convenience': 'grocery', 'greengrocer': 'grocery',
    'kiosk': 'grocery', 'general': 'grocery', 'bakery': 'grocery', 'butcher': 'grocery',
    'chemist': 'pharmacy', 'pharmacy': 'pharmacy', 'medical_supply': 'pharmacy', 'optician': 'pharmacy',
    'mall': 'mall', 'department_store': 'mall', 'clothes': 'mall', 'shoes': 'mall',
    'electronics': 'mall', 'jewelry': 'mall', 'cosmetics': 'mall', 'sports': 'mall',
    'furniture': 'mall', 'bag': 'mall', 'mobile_phone': 'mall', 'gift': 'mall',
    'antiques': 'market', 'variety_store': 'market', 'second_hand': 'market',
}
_AMENITY_TO_KIND = {
    'restaurant': 'restaurant', 'bar': 'restaurant', 'pub': 'restaurant', 'fast_food': 'restaurant',
    'food_court': 'restaurant', 'biergarten': 'restaurant', 'cafe': 'cafe', 'ice_cream': 'cafe',
    'pharmacy': 'pharmacy', 'clinic': 'pharmacy', 'doctors': 'pharmacy', 'dentist': 'pharmacy',
    'bank': 'lobby', 'post_office': 'lobby', 'townhall': 'lobby',
}


def interior_kind_for_tags(tags, default='home'):
    """Map OSM tags to an INTERIOR_KINDS id (a stable, data-driven kind for a unit)."""
    tags = tags or {}
    name = None
    if tags.get('shop'):
        name = _SHOP_TO_KIND.get(tags['shop'], 'retail')
    elif tags.get('amenity') in _AMENITY_TO_KIND:
        name = _AMENITY_TO_KIND[tags['amenity']]
    elif tags.get('leisure') in ('fitness_centre', 'sports_centre', 'fitness_station'):
        name = 'gym'
    elif tags.get('tourism') in ('hotel', 'motel', 'hostel', 'guest_house', 'apartment'):
        name = 'hotel'
    elif tags.get('office'):
        name = 'office'
    elif tags.get('building') in ('retail', 'commercial', 'mall', 'supermarket'):
        name = 'mall' if tags.get('building') == 'mall' else 'retail'
    elif tags.get('building') in ('office',):
        name = 'office'
    elif tags.get('building') in ('hotel',):
        name = 'hotel'
    return INTERIOR_KINDS.index(name or default)


INTERIORS = """
// Room coordinates are metres: X along the facade, Y up, Z into the building.
// Only the facade writes scene depth; the nearest solid inside it determines
// visibility. Furniture never inherits a panorama's perspective or curvature.
const int IR_WOOD = 1, IR_METAL = 2, IR_FABRIC = 3, IR_BOOKS = 4;
const int IR_SCREEN = 5, IR_CABINET = 6, IR_LINEN = 7, IR_PILLOW = 8;
const int IR_POT = 9, IR_LEAVES = 10, IR_SHADE = 11, IR_STOCK = 12;
const int IR_PLASTER = 13, IR_FRIDGE = 14, IR_SIGN = 15, IR_CLOTHES = 16;
const int IR_RUBBER = 17, IR_COUNTER = 18;

// Room kinds (keep in sync with INTERIOR_KINDS).
const int IK_CAFE = 0, IK_RETAIL = 1, IK_MARKET = 2, IK_ARCADE = 3, IK_LOBBY = 4;
const int IK_OFFICE = 5, IK_HOME = 6, IK_MALL = 7, IK_GROCERY = 8, IK_RESTAURANT = 9;
const int IK_PHARMACY = 10, IK_GYM = 11, IK_HOTEL = 12;

bool interiorIsRowLit(int kind) {
    // Long linear fixtures running into the room (shop and gym lighting).
    return kind == IK_MALL || kind == IK_GROCERY || kind == IK_PHARMACY || kind == IK_GYM;
}
bool interiorIsShop(int kind) { return interiorIsRowLit(kind); }
int interiorVariantCount(int kind) { return kind == IK_HOME ? 8 : kind >= IK_OFFICE ? 6 : 3; }
float interiorMinWidth(int kind) {
    if (kind == IK_MALL || kind == IK_GYM) return 4.6;
    if (kind == IK_GROCERY || kind == IK_PHARMACY || kind == IK_RESTAURANT) return 4.0;
    return kind < IK_OFFICE ? 3.6 : 2.7;
}
float interiorMinDepth(int kind) { return kind >= IK_MALL && kind != IK_HOTEL ? 4.2 : 3.1; }
// Practical lighting colour and brightness differ by use: cool tubes in groceries and
// pharmacies, warm dim light in restaurants and hotels, neutral bright light in malls.
vec3 interiorKindTint(int kind) {
    if (kind == IK_GROCERY || kind == IK_PHARMACY) return vec3(.90, 1.0, 1.10);
    if (kind == IK_GYM) return vec3(.95, 1.0, 1.06);
    if (kind == IK_MALL) return vec3(1.04, 1.0, .93);
    if (kind == IK_RESTAURANT) return vec3(1.20, .93, .62);
    if (kind == IK_HOTEL) return vec3(1.12, .97, .74);
    return vec3(1.0);
}
float interiorKindGain(int kind) {
    if (kind == IK_GROCERY) return 1.35;
    if (kind == IK_PHARMACY) return 1.30;
    if (kind == IK_MALL) return 1.20;
    if (kind == IK_GYM) return 1.10;
    if (kind == IK_RESTAURANT) return .55;
    if (kind == IK_HOTEL) return .70;
    return 1.0;
}
float interiorFloorGloss(int kind) {
    if (kind == IK_MALL) return .95;
    if (kind == IK_PHARMACY) return .80;
    if (kind == IK_GROCERY) return .60;
    return 0.0;
}

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
    int count = interiorVariantCount(kind);
    return min(int(floor(h.y * float(count))), count - 1);
}
void interiorLamps(vec3 size, int kind, vec4 h, out vec3 a, out vec3 b) {
    float drop = kind == IK_CAFE || kind == IK_HOME || kind == IK_RESTAURANT ? .62 : .055;
    a = vec3(size.x * (.28 + .08 * h.x), size.y - drop, size.z * (.30 + .08 * h.w));
    b = vec3(size.x * (.68 + .08 * h.w), size.y - drop, size.z * (.63 + .08 * h.x));
}
""" + INTERIOR_LAYOUTS + INTERIOR_STORES + """

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

// Ceiling light strips run into the room; each point is lit by its nearest strip.
float interiorStripCount(vec3 size) { return clamp(floor(size.x / 2.4), 1.0, 5.0); }
float interiorStripHalf(int kind) { return kind == IK_MALL ? .045 : kind == IK_GYM ? .05 : .14; }
float interiorStripMask(vec2 xz, vec3 size, int kind, float pad, float fw) {
    float cell = size.x / interiorStripCount(size);
    vec2 p = vec2(mod(xz.x, cell) - cell * .5, xz.y);
    float hw = interiorStripHalf(kind) + pad;
    return roomRect(p, vec2(-hw, .45 - pad), vec2(hw, size.z - .45 + pad), fw);
}
vec3 interiorRowLight(vec3 P, vec3 size) {
    float n = interiorStripCount(size), cell = size.x / n;
    float x = (clamp(floor(P.x / cell), 0.0, n - 1.0) + .5) * cell;
    return vec3(x, size.y - .04, clamp(P.z, .45, size.z - .45));
}

vec3 interiorBrand(vec4 h) {
    float k = fract(h.w * 3.7 + h.x * 1.9);
    if (k < .17) return vec3(.70, .15, .13);
    if (k < .34) return vec3(.13, .44, .27);
    if (k < .51) return vec3(.14, .29, .56);
    if (k < .68) return vec3(.86, .47, .12);
    if (k < .84) return vec3(.12, .50, .52);
    return vec3(.42, .20, .48);
}
vec3 interiorKindWall(int kind, vec4 h) {
    if (kind == IK_MALL) return mix(vec3(.86, .83, .77), vec3(.74, .70, .66), step(.6, h.x));
    if (kind == IK_GROCERY) return vec3(.88, .88, .85);
    if (kind == IK_PHARMACY) return vec3(.90, .91, .89);
    if (kind == IK_GYM) return h.x < .75 ? vec3(.62, .63, .62) : vec3(.27, .28, .30);
    if (kind == IK_RESTAURANT)
        return h.x < .35 ? vec3(.20, .29, .24) : h.x < .70 ? vec3(.56, .33, .25) : vec3(.80, .74, .62);
    if (kind == IK_HOTEL) return h.x < .5 ? vec3(.74, .70, .63) : vec3(.62, .62, .60);
    return interiorWall(h);
}

// Planar wall details specific to a use: signage, mirrors, wainscot, brand stripes.
vec3 interiorWallDetail(vec3 alb, vec2 p, float width, vec3 N, vec3 size, int kind,
                        vec4 h, vec3 brand, float fw, out float fixture) {
    fixture = 0.0;
    bool back = N.z < -.5;
    float top = size.y;
    if (kind == IK_GROCERY) {
        float y0 = top - .78;
        alb = mix(alb, brand, roomRect(p, vec2(-1.0, y0), vec2(width + 1.0, y0 + .26), fw));
    } else if (kind == IK_PHARMACY) {
        float y0 = top - .72;
        alb = mix(alb, vec3(.19, .56, .33), roomRect(p, vec2(-1.0, y0), vec2(width + 1.0, y0 + .12), fw));
        if (back) {
            vec2 q = p - vec2(width * (h.w < .5 ? .62 : .35), min(2.15, top - .58));
            float cross = max(roomRect(q, vec2(-.05, -.17), vec2(.05, .17), fw),
                              roomRect(q, vec2(-.17, -.05), vec2(.17, .05), fw));
            alb = mix(alb, vec3(.10), roomRect(q, vec2(-.25), vec2(.25), fw) * .85);
            alb = mix(alb, vec3(.15, .80, .38), cross);
            fixture = cross * .55;
        }
    } else if (kind == IK_GYM) {
        if (back || (h.w > .5 && N.x < -.5)) {
            float mh = min(2.45, top - .30);
            float m = roomRect(p, vec2(.35, .14), vec2(width - .35, mh), fw);
            float seam = band(p.x / 1.15, .012, .988, fw / 1.15);
            vec3 glass = vec3(.40, .46, .48) * (.62 + .38 * smoothstep(.0, mh, p.y));
            float glare = band((p.x - p.y * .75) / 2.6, .40, .56, fw / 2.6) * .22;
            alb = mix(alb, (glass + glare) * (.78 + .22 * seam), m);
            fixture = m * .012;
        }
    } else if (kind == IK_RESTAURANT) {
        float wain = 1.0 - smoothstep(1.05 - fw, 1.05 + fw, p.y);
        float panel = band(p.x / .62, .035, .965, fw / .62);
        alb = mix(alb, interiorWood(h) * .62 * (.82 + .18 * panel), wain * .92);
        alb = mix(alb, interiorWood(h) * .82, roomRect(p, vec2(-1.0, 1.05), vec2(width + 1.0, 1.11), fw));
        if (back) {
            vec2 q = p - vec2(width * (h.w < .5 ? .64 : .34), 1.62);
            float board = roomRect(q, vec2(-.75, -.45), vec2(.75, .45), fw);
            float frame = roomRect(q, vec2(-.80, -.50), vec2(.80, .50), fw);
            float row = floor((q.y + .45) * 7.0);
            float lines = band((q.y + .45) * 7.0, .25, .55, fw * 7.0)
                        * roomRect(q, vec2(-.62, -.40), vec2(.62 - .5 * fract(row * .37), .40), fw);
            alb = mix(alb, interiorWood(h) * .40, frame);
            alb = mix(alb, mix(vec3(.06, .07, .065), vec3(.62, .62, .58), lines * .8), board);
        }
    } else if (kind == IK_MALL) {
        if (back) {
            float sx0 = width * .22, sx1 = width * .78, y0 = top - .98, y1 = top - .32;
            float box = roomRect(p, vec2(sx0, y0), vec2(sx1, y1), fw);
            float inner = roomRect(p, vec2(sx0 + .06, y0 + .06), vec2(sx1 - .06, y1 - .06), fw);
            vec2 q = vec2((p.x - sx0) / (sx1 - sx0), (p.y - y0) / (y1 - y0));
            float letters = band(q.x * 11.0, .15, .70, fw * 11.0 / (sx1 - sx0))
                          * roomRect(q, vec2(.12, .28), vec2(.88, .72), fw * 4.0);
            alb = mix(alb, vec3(.12), box);
            alb = mix(alb, mix(brand, vec3(.93), letters), inner);
            fixture = inner * (.18 + .32 * letters);
        } else if (abs(N.x) > .5 && h.w > .4) {
            float slat = band(p.x / .10, .12, .88, fw / .10);
            float lower = 1.0 - smoothstep(top - .60 - fw, top - .60 + fw, p.y);
            alb = mix(alb, interiorWood(h) * (.55 + .35 * slat), lower * .95);
        }
    } else if (kind == IK_HOTEL) {
        alb *= .955 + .045 * band(p.x / .16, .40, .60, fw / .16);
    }
    return alb;
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
    vec3 wall = interiorKindWall(kind, h), wood = interiorWood(h), alb = wall;
    vec3 brand = interiorBrand(h);
    int variant = interiorVariant(kind, h);
    bool shop = interiorIsShop(kind);
    fixture = 0.0;
    float fine = 1.0 - smoothstep(.035, .16, fw);
    bool office = kind == 5;
    if (N.y > .5) {
        if (kind == IK_HOME && variant == 6) {
            // Vacant unit: poured screed with a faint trowel mottling.
            float mott = .5 + .5 * sin(P.x * 3.1 + sin(P.z * 2.3) * 2.0);
            alb = vec3(.52, .51, .49) * (.92 + .08 * mott) * (.96 + .08 * mix(.5, hash21(floor(P.xz * 40.0)), fine));
        } else if (office && h.w < .70) {
            vec2 id = floor(P.xz / .60);
            float tile = mix(1.0, .90 + .15 * hash21(id + h.z * 19.0), fine);
            float seam = band(P.x / .60, .012, .988, fw / .60) * band(P.z / .60, .012, .988, fw / .60);
            alb = mix(vec3(.31, .33, .32), vec3(.39, .36, .31), h.x) * tile * (.9 + .1 * seam);
        } else if (kind == IK_MALL) {
            // Polished stone concourse tiles with a darker inlay band.
            float seam = band(P.x / 1.2, .008, .992, fw / 1.2) * band(P.z / 1.2, .008, .992, fw / 1.2);
            vec2 id = floor(P.xz / 1.2);
            vec3 stone = mix(vec3(.74, .70, .64), vec3(.60, .58, .55), h.w);
            alb = stone * mix(1.0, .90 + .14 * hash21(id + h.x * 7.0), fine) * (.84 + .16 * seam);
        } else if (kind == IK_GROCERY || kind == IK_PHARMACY) {
            float sz = kind == IK_GROCERY ? .40 : .60;
            float seam = band(P.x / sz, .015, .985, fw / sz) * band(P.z / sz, .015, .985, fw / sz);
            float check = mod(floor(P.x / sz) + floor(P.z / sz), 2.0);
            vec3 base = kind == IK_GROCERY ? vec3(.74, .73, .69) : vec3(.80, .81, .79);
            alb = base * (1.0 - .06 * check * fine) * (.85 + .15 * seam);
        } else if (kind == IK_GYM) {
            float speck = hash21(floor(P.xz * 55.0));
            alb = vec3(.13, .13, .135) * mix(1.0, .75 + .9 * step(.93, speck), fine)
                * (.9 + .1 * band(P.x / 1.0, .01, .99, fw));
        } else if (kind == IK_HOTEL) {
            float n = hash21(floor(P.xz * 90.0));
            float stripe = band(P.x / .46, .35, .65, fw / .46);
            vec3 carpet = h.x < .5 ? vec3(.30, .27, .25) : vec3(.27, .29, .31);
            alb = carpet * mix(1.0, .90 + .20 * n, fine) * (.93 + .07 * stripe);
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
            vec3 timber = kind == 0 ? mix(wood, vec3(.65, .59, .49), .60)
                        : kind == IK_RESTAURANT ? wood * .78 : wood;
            float grain = sin(P.x * 175.0 + sin(P.z * 1.4 + row) * 1.8);
            float grainFade = 1.0 - smoothstep(.005, .035, fw);
            alb = timber * mix(1.0, .91 + .17 * board, fine) * (.94 + .06 * seam)
                  * (1.0 + .025 * grain * grainFade);
        }
        // Woven rugs anchor seating and beds; their detail is flat on the floor.
        if ((kind == 6 && variant != 6) || (kind == 0 && variant == 0)) {
            vec2 halfRug = size.xz * vec2(.32, .28);
            vec2 at = P.xz - size.xz * vec2(.49, .45);
            float rug = roomRect(at, -halfRug, halfRug, fw);
            float border = roomRect(at, -halfRug + .12, halfRug - .12, fw);
            vec3 cloth = mix(interiorCloth(h) * .72, mix(interiorCloth(h), vec3(.70, .64, .52), .45), border);
            alb = mix(alb, cloth, rug * .90);
        }
    } else if (N.y < -.5) {
        alb = vec3(.86, .85, .81);
        if (kind == IK_GYM) {
            // Exposed soffit with ducting beams.
            float beam = 1.0 - band(P.x / 1.8 + h.w, .04, .96, fw / 1.8);
            alb = mix(vec3(.30, .30, .31), vec3(.16, .16, .17), beam * .7);
        } else if (kind == IK_RESTAURANT) {
            float slat = band(P.x / .12, .08, .92, fw / .12);
            alb = mix(vec3(.17, .15, .13), vec3(.30, .23, .17), slat * .55);
        } else if (office || kind == 2 || kind == 3 || kind == IK_GROCERY || kind == IK_PHARMACY) {
            float grid = band(P.x / .60, .009, .991, fw / .60) * band(P.z / .60, .009, .991, fw / .60);
            alb *= .89 + .11 * grid;
        } else if (kind != 0 && kind < IK_MALL) {
            float beam = 1.0 - band(P.z / 2.15 + h.w, .035, .965, fw / 2.15);
            alb = mix(alb, wall * .78, beam * .55);
        }
        if (interiorIsRowLit(kind)) {
            fixture = interiorStripMask(P.xz, size, kind, 0.0, fw);
            alb *= 1.0 - .22 * interiorStripMask(P.xz, size, kind, .035, fw);
        } else if (kind == IK_HOTEL) {
            vec3 a, b; interiorLamps(size, kind, h, a, b);
            float ra = 1.0 - smoothstep(.085 - fw, .085 + fw, length(P.xz - a.xz));
            float rb = 1.0 - smoothstep(.085 - fw, .085 + fw, length(P.xz - b.xz));
            fixture = max(ra, rb);
        } else if (kind != 0 && kind != 6 && kind != IK_RESTAURANT) {
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
        bool accent = kind == 0 || (kind == 1 && h.x < .5) || (kind == IK_RESTAURANT && h.w < .45);
        if (accent && (N.x < -.5 || N.z < -.5)) {
            vec2 brick = p / vec2(.28, .085);
            brick.x += .5 * mod(floor(brick.y), 2.0);
            float inside = band(brick.x, .026, .974, fw / .28) * band(brick.y, .07, .93, fw / .085);
            vec3 clay = mix(vec3(.58, .42, .31), vec3(.73, .66, .52), h.x);
            clay *= mix(1.0, .92 + .15 * hash21(floor(brick)), fine);
            alb = mix(vec3(.66, .63, .54), clay, inside);
        }
        if (P.y < .13) alb = shop ? vec3(.24, .24, .25) : wood * .56;
        float cornice = smoothstep(size.y - .14 - fw, size.y - .14 + fw, P.y);
        alb = mix(alb, wall * .85, cornice);
        alb = interiorWallDetail(alb, p, width, N, size, kind, h, brand, fw, fixture);
        // Back doors, elevator panels, artwork and wall trim are planar details.
        float dx = width * mix(.20, .78, step(.5, h.w));
        if (N.z < -.5) {
            float door = roomRect(p, vec2(dx - .45, .13), vec2(dx + .45, 2.25), fw);
            float inner = roomRect(p, vec2(dx - .40, .14), vec2(dx + .40, 2.19), fw);
            vec3 doorCol = kind == 4 || shop ? vec3(.53, .55, .53) : wood * .77;
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
            float present = step(.22, hash21(vec2(index + h.x * 5.0, N.x + h.y * 8.0))) * (shop ? 0.0 : 1.0);
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
    else if (m == IR_RUBBER) alb = vec3(.075, .075, .08);
    else if (m == IR_COUNTER) alb = h.x < .6 ? vec3(.84, .83, .80) : vec3(.20, .21, .22);
    else if (m == IR_CLOTHES) {
        // Hanging garments read as narrow vertical colour bands; mannequins are one colour.
        float u = s.z <= s.x ? q.x : q.z;
        float pitch = hit.shape == 0 ? .05 : 10.0;
        float r = hash21(vec2(floor(u / pitch), floor(h.x * 7.0) + hit.center.x));
        float r2 = fract(r * 7.13);
        vec3 c = interiorCloth(vec4(r, 0.0, 0.0, r2));
        c = mix(c, interiorBrand(vec4(r2, 0.0, 0.0, r)), step(.55, r));
        c = mix(c, vec3(.86), step(.9, r2) * .8);
        alb = mix(c * .8, c * mix(.80, 1.0, band(u / pitch, .08, .92, fw / pitch)), fine);
    } else if (m == IR_SIGN) {
        vec3 c = interiorBrand(vec4(fract(hit.center.x * .31), 0.0, 0.0, fract(hit.center.z * .47)));
        vec2 face = q.xy;
        float inset = roomRect(face, -s.xy + .03, s.xy - .03, fw) * abs(hit.n.z);
        float txt = band(face.x * 5.0, .15, .65, fw * 5.0)
                  * roomRect(face, vec2(-s.x * .8, -s.y * .45), vec2(s.x * .8, s.y * .45), fw);
        alb = mix(vec3(.15), mix(c, vec3(.95), txt), inset);
        fixture = inset * (.30 + .25 * txt);
    } else if (m == IR_SHADE) {
        float cage = band(atan(q.z, q.x) * 3.18, .10, .87, fw / max(s.x, .05) * 3.18);
        float rim = smoothstep(s.y - .035 - fw, s.y - .035 + fw, abs(q.y));
        alb = mix(vec3(.18, .17, .13), vec3(.84, .78, .59), cage * (1.0 - rim));
        fixture = (hit.n.y < -.5 ? .9 : .58 * cage) * (1.0 - rim * .8);
    } else if (m == IR_BOOKS || m == IR_STOCK || m == IR_CABINET || m == IR_SCREEN || m == IR_FRIDGE) {
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
            float pitch = m == IR_BOOKS ? .105 : m == IR_FRIDGE ? .12 : .16;
            vec2 id = floor(uv / vec2(pitch, .39));
            vec2 cell = fract(uv / vec2(pitch, .39));
            float r = hash21(id + hit.center.xz + h.w * 13.0);
            float stock = band(uv.x / pitch, .07, .92, fw / pitch)
                        * band(uv.y / .39, .12, mix(.62, .93, r), fw / .39);
            vec3 color = mix(interiorCloth(h), mix(vec3(.71, .55, .35), vec3(.50, .57, .57), r), .62);
            // Packaged goods are brightly branded; books stay muted.
            if (m != IR_BOOKS)
                color = mix(color, interiorBrand(vec4(r, 0.0, 0.0, fract(r * 13.7 + id.y * .31))), .50);
            float label = roomRect(cell, vec2(.2, .30), vec2(.78, .42), fw / pitch);
            color = mix(color, vec3(.78, .75, .65), label * .6);
            vec3 contents = mix(interiorWood(h) * .23, color, mix(.65, stock, fine));
            alb = mix(alb, contents, inset);
            if (m == IR_FRIDGE) {
                // Glass-door chiller: lit interior, door mullions every ~0.78 m.
                float door = band(uv.x / .78, .012, .988, fw / .78);
                alb = mix(vec3(.58, .60, .60), contents * (.80 + .20 * door), inset);
                fixture = inset * door * .20;
            }
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
    // Use-specific lamp colour and brightness; a vacant flat is mostly dark.
    lampCol *= interiorKindTint(kind);
    lit *= interiorKindGain(kind);
    if (kind == IK_HOME && interiorVariant(kind, h) == 6) lit *= .10;
    vec3 mean = linearColor(interiorKindWall(kind, h)) * .49 + linearColor(interiorWood(h)) * .19
              + linearColor(vec3(.84, .83, .79)) * .23;
    // Furniture, cabinetry and floor coverings between the ceiling lamps and the glass
    // absorb a large share of the room light, and part of every pane carries cloth.
    // Averaging a room down to its brightest surfaces is what turns a distant suite
    // into one uniform glowing slab: the window then reads far larger and brighter than
    // the lit slot behind the furniture actually is.  Fold that occlusion into the far
    // average so an unresolved pane stays a dim, uneven light.
    float furnish = mix(.58, .82, h.y);
    float curtain = mix(.12, .36, step(.55, h.w));
    if (kind >= IK_MALL) curtain *= .45;
    vec3 flatRad = interiorLight(mean * furnish, .70, interiorIsRowLit(kind) ? .032 : .010,
                                 lit, lampCol, amb * .40);
    vec3 lamp = lampCol / max(dot(lampCol, PHOTO_LUMA), 1e-5);
    vec3 cloth = linearColor(mix(vec3(.80, .77, .70), vec3(.60, .62, .62), h.x))
               * (amb * 1.2 + lamp * lit * interiorSurfaceGain() * .58);
    flatRad = mix(flatRad, cloth, curtain);
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

    // Glossy floors mirror the ceiling strips: a blurred, Fresnel-weighted streak.
    float gloss = hit.material == 0 && N.y > .5 ? interiorFloorGloss(kind) : 0.0;
    if (gloss > 0.0) {
        vec3 rd = vec3(ray.d.x, -ray.d.y, ray.d.z);
        float tc = (rs.y - P.y) / max(rd.y, .02);
        vec3 Q = P + rd * tc;
        float f = .035 + .965 * pow(1.0 - clamp(rd.y, 0.0, 1.0), 5.0);
        float inside = step(0.0, Q.x) * step(Q.x, rs.x) * step(0.0, Q.z) * step(Q.z, rs.z);
        fixture = max(fixture, interiorStripMask(Q.xz, rs, kind, 0.0, max(fw, .05 + .07 * tc))
                               * f * gloss * inside * .9);
    }

    float direct;
    if (interiorIsRowLit(kind)) {
        vec3 va = interiorRowLight(P, rs) - P;
        float aa = dot(va, va);
        direct = 2.1 * max(dot(N, va * inversesqrt(max(aa, .0001))), 0.0) / (1.0 + .22 * aa);
    } else {
        vec3 a, b; interiorLamps(rs, kind, h, a, b);
        vec3 va = a - P, vb = b - P;
        float aa = dot(va, va), bb = dot(vb, vb);
        direct = max(dot(N, va * inversesqrt(max(aa, .0001))), 0.0) / (1.0 + .22 * aa)
               + max(dot(N, vb * inversesqrt(max(bb, .0001))), 0.0) / (1.0 + .22 * bb);
    }
    float bounce = mix(.38, .68, max(-N.y, 0.0)) * (1.0 - .18 * P.z / rs.z);
    float diffuse = (direct * 1.65 + bounce) * ao;
    float windowLight = (.32 + 1.05 * exp(-P.z / 3.8)) * (.45 + .55 * max(-N.z, 0.0));
    vec3 rad = interiorLight(albedo, diffuse, fixture, lit, lampCol, amb * windowLight * ao);
    return mix(rad, flatRad, lod);
}

// kind is one of IK_* (see INTERIOR_KINDS / interior_kind_for_tags in Python).
vec3 interiorRadianceKind(vec3 d, vec2 o, vec2 rs, float D, vec4 h, float lit,
                           vec3 lampCol, vec3 amb, int kind, vec2 footprint) {
    return furnishedRoom(d, o, rs, D, kind, h, lit, lampCol, amb, footprint);
}
// Historical entry point: still available for callers that only know office vs home.
vec3 interiorRadiance(vec3 d, vec2 o, vec2 rs, float D, vec4 h, float lit,
                       vec3 lampCol, vec3 amb, bool office, vec2 footprint) {
    return interiorRadianceKind(d, o, rs, D, h, lit, lampCol, amb, office ? IK_OFFICE : IK_HOME, footprint);
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

// Kind-aware dressing: office / home keep the original blinds and curtains; the other
// uses add what real frontages have (vinyl, frosting, cafe curtains, hotel drapes).
vec3 windowDressingKind(vec3 rad, vec3 d, vec2 pane, vec2 baySize, vec4 rect,
                         vec4 h, float lit, vec3 lampCol, vec3 amb, int kind) {
    if (kind == IK_OFFICE) return windowDressing(rad, d, pane, baySize, rect, h, lit, lampCol, amb, true);
    if (kind < IK_MALL) return windowDressing(rad, d, pane, baySize, rect, h, lit, lampCol, amb, false);
    vec2 p = (pane + d.xy / max(d.z, .04) * .18 / baySize - rect.xz)
           / max(rect.yw - rect.xz, vec2(.05));
    vec2 fw = max(fwidth(p), vec2(.001));
    lampCol *= interiorKindTint(kind);
    vec3 lamp = lampCol / max(dot(lampCol, PHOTO_LUMA), 1e-5);
    vec3 fill = amb * 1.3 + lamp * lit * interiorKindGain(kind) * interiorSurfaceGain() * .62;
    float detail = 1.0 - smoothstep(.008, .05, fw.x);
    float phase = p.x * baySize.x * (rect.y - rect.x) * 42.0 + .3 * sin(p.y * 2.6 + h.z * 5.0);
    float folds = .82 + (.11 * cos(phase) + .03 * cos(phase * 2.0 + .4)) * detail;
    if (kind == IK_HOTEL) {
        // Drapes either side plus a sheer net across the glass.
        float width = mix(.16, .30, h.x);
        float cover = max(1.0 - smoothstep(width - fw.x, width + fw.x, p.x),
                          smoothstep(1.0 - width - fw.x, 1.0 - width + fw.x, p.x)) * step(.12, h.w);
        cover *= smoothstep(.02 - fw.y, .02 + fw.y, p.y);
        vec3 drape = linearColor(mix(vec3(.58, .52, .46), vec3(.40, .43, .46), h.x));
        vec3 net = linearColor(vec3(.88, .87, .82)) * fill;
        rad = mix(rad, net, step(.40, h.y) * (1.0 - cover) * .20);
        return mix(rad, drape * fill * folds, cover * .96);
    }
    if (kind == IK_RESTAURANT) {
        // Cafe curtain on a rod across the lower part of the window.
        float rod = mix(.20, .34, h.z);
        float cover = (1.0 - smoothstep(rod - fw.y, rod + fw.y, p.y)) * step(.45, h.w);
        vec3 cloth = linearColor(mix(vec3(.86, .82, .72), vec3(.70, .74, .70), h.x));
        return mix(rad, cloth * fill * folds, cover * .93);
    }
    if (kind == IK_GROCERY || kind == IK_PHARMACY) {
        // Frosted kick band and promotional decals; pharmacies always carry the band.
        float frost = (1.0 - smoothstep(.07 - fw.y, .07 + fw.y, p.y))
                    * (kind == IK_PHARMACY ? 1.0 : step(.35, h.y));
        vec3 brand = linearColor(interiorBrand(h));
        float x0 = mix(.05, .50, h.x);
        float decal = max(roomRect(p, vec2(x0, .18), vec2(x0 + .07, .46), fw.x),
                          roomRect(p, vec2(x0 + .10, .26), vec2(x0 + .16, .50), fw.x)) * step(.45, h.w);
        rad = mix(rad, linearColor(vec3(.86, .88, .87)) * fill, frost * .70);
        return mix(rad, brand * fill * 1.2, decal * .88);
    }
    if (kind == IK_GYM) {
        // A mid-height logo / privacy stripe.
        float stripe = roomRect(p, vec2(-1.0, .40), vec2(2.0, .49), fw.y) * step(.30, h.w);
        float logo = roomRect(p, vec2(.08, .415), vec2(.22, .475), fw.x);
        vec3 brand = linearColor(interiorBrand(h));
        rad = mix(rad, linearColor(vec3(.82, .84, .84)) * fill, stripe * .62);
        return mix(rad, brand * fill, logo * stripe * .85);
    }
    return rad;   // mall units stay open glass
}
"""