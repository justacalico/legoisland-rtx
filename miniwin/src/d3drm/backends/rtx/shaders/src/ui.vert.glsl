#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec2 uv;

void main()
{
	// draw.modelView already contains the composed 2D transform * ortho matrix.
	gl_Position = draw.modelView * vec4(inPos, 1.0);
	uv = inUV;
}
