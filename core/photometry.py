"""Reference photometry for the scene's normalized linear-light units.

100,000 lux direct sunlight is 2.35 scene irradiance units.  Lambertian
surfaces return reflectance * irradiance / pi; emitters use cd/m² (nits).
These are representative urban fixtures, not measurements of individual OSM assets.
"""

SUN_LUX = 100_000.0
SUN_IRRADIANCE = 2.35
NITS_PER_SCENE_UNIT = SUN_LUX / SUN_IRRADIANCE
MOON_LUX = 0.25
NIGHT_EXPOSURE_STOPS = 10.0
DAY_EXPOSURE = 1.1
STREET_LAMP_CANDELA = 3000.0  # peak of a broad downward roadway distribution
HEADLIGHT_CANDELA = 12000.0   # low-beam projector, aimed a few degrees below the horizon
OFFICE_LUX = 750.0
INTERIOR_DAY_GAIN = 3.2  # room surfaces seen through glass in daylight sit ~1:50 below a sunlit facade
INTERIOR_NIGHT_GAIN = 0.08  # subdued occupied rooms under the ten-stop night exposure
INTERIOR_FIXTURE_DAY_GAIN = 1.1  # ceiling fixtures are faint by day: ~1:10 of a sunlit wall
INTERIOR_FIXTURE_NIGHT_GAIN = 0.012  # compensate for night exposure without clipping whole rooms

LUMINANCE_NITS = {
    "SIGN": 150.0,
    "SCREEN_DAY": 500.0,
    "SCREEN_NIGHT": 120.0,
    "SHOP": 80.0,
    "CROWN": 40.0,
    "BEACON": 2000.0,
    "WINDOW": 40.0,
    "CEILING": 2500.0,
    "STREET_LAMP": 5000.0,
    "HEADLIGHT": 15500.0,
    "TAILLIGHT": 1150.0,
    "BRAKELIGHT": 6500.0,
    "INDICATOR": 8000.0,
    "TRAFFIC_SIGNAL": 8000.0,
    "DRL": 4000.0,
}

# Linear diffuse reflectance ranges; weathering may darken these further.
REFLECTANCE = {
    "ASPHALT": (0.02, 0.08),
    "CONCRETE": (0.20, 0.51),
    "BRICK": (0.08, 0.35),
    "PLASTER": (0.35, 0.80),
    "SOIL": (0.08, 0.20),
    "FOLIAGE": (0.05, 0.21),
    "ROOF": (0.08, 0.35),
    "PAINT": (0.02, 0.80),
}
