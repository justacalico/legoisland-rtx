#version 460

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;

layout(push_constant) uniform PostParams {
	vec4 params; // x = exposure, y = bloom amount, z = apply gamma, w = dither
} post;

// ACES filmic tonemapping (Narkowicz approximation).
vec3 acesTonemap(vec3 x)
{
	const float a = 2.51;
	const float b = 0.03;
	const float c = 2.43;
	const float d = 0.59;
	const float e = 0.14;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float hash12(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

void main()
{
	vec3 hdr = texture(hdrTex, uv).rgb;

	// Cheap single-pass bloom: poisson taps on a bright-passed neighborhood.
	if (post.params.y > 0.0) {
		vec3 bloom = vec3(0.0);
		vec2 texel = 1.0 / vec2(textureSize(hdrTex, 0));
		for (int i = 0; i < 8; ++i) {
			float ang = float(i) * 0.785398;
			vec2 off = vec2(cos(ang), sin(ang)) * texel * 6.0;
			vec3 s = texture(hdrTex, uv + off).rgb;
			bloom += max(s - vec3(1.0), vec3(0.0));
		}
		hdr += (bloom / 8.0) * post.params.y;
	}

	hdr *= post.params.x;
	vec3 ldr = acesTonemap(hdr);

	if (post.params.z != 0.0) {
		ldr = pow(ldr, vec3(1.0 / 2.2));
	}

	// Ordered-ish dither to kill banding on dark gradients.
	if (post.params.w != 0.0) {
		ldr += (hash12(gl_FragCoord.xy) - 0.5) / 255.0;
	}

	outColor = vec4(ldr, 1.0);
}
