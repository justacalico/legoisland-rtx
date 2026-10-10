#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 uv;

void main()
{
	vec4 viewPos = draw.modelView * vec4(inPos, 1.0);
	gl_Position = frame.proj * viewPos;
	worldPos = (draw.world * vec4(inPos, 1.0)).xyz;
	worldNormal = normalize(mat3(draw.normal) * inNormal);
	uv = inUV;
}
