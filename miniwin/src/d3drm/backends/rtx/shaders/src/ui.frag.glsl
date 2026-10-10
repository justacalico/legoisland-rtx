#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 1) uniform sampler2D tex;

void main()
{
	vec4 texel = texture(tex, uv);
	outColor = vec4(texel.rgb * draw.color.rgb, texel.a * draw.color.a);
}
