"""Scene depth decoding; shadow maps retain conventional OpenGL depth."""

SCENE_DEPTH = """
uniform float uReverseZ;
float depthNdc(float d) { return uReverseZ > .5 ? d : d * 2.0 - 1.0; }
bool depthSky(float d) { return uReverseZ > .5 ? d <= 0.0 : d >= 1.0; }
"""
