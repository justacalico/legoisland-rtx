#version 460

#extension GL_GOOGLE_include_directive : require

#ifdef RTX_RT
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#endif

#include "common.glsl"

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 1) uniform sampler2D tex;

#ifdef RTX_RT
layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;

struct RTInstance {
	uvec2 vb;
	uvec2 ib;
	vec4 color;
	uvec4 params; // x = index count, y = flags, z = texture id, w = unused
};

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuf {
	RTInstance inst[];
};

layout(buffer_reference, std430) readonly buffer RTVerts {
	vec4 v[]; // 2 vec4 per vertex: [pos.xyz|norm.x][norm.yz|uv.xy]
};
layout(buffer_reference, std430) readonly buffer RTIdx {
	uint i[];
};

uint fetchIndex(RTIdx ib, uint i)
{
	uint w = ib.i[i >> 1];
	return (w >> ((i & 1u) * 16u)) & 0xFFFFu;
}

vec3 fetchPos(RTVerts vb, uint i)
{
	return vb.v[i * 2].xyz;
}

vec3 fetchNormal(RTVerts vb, uint i)
{
	vec4 a = vb.v[i * 2];
	vec4 b = vb.v[i * 2 + 1];
	return vec3(a.w, b.x, b.y);
}

// Returns true if any opaque geometry is hit along the ray.
bool rayOccluded(vec3 origin, vec3 dir, float tmin, float tmax, uint mask)
{
	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, tlas, gl_RayFlagsTerminateOnFirstHitEXT, mask, origin, tmin, dir, tmax);
	while (rayQueryProceedEXT(rq)) {
	}
	return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// Shades a committed triangle hit: fetches geometry through buffer references
// and applies the scene lights with plain diffuse + the material color.
vec3 shadeRayHit(rayQueryEXT rq, vec3 rayDir)
{
	uint customIdx = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
	uint prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
	vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
	float t = rayQueryGetIntersectionTEXT(rq, true);

	RTInstance info = inst[customIdx];
	RTVerts verts = RTVerts(info.vb);
	RTIdx idx = RTIdx(info.ib);

	uint i0 = fetchIndex(idx, prim * 3);
	uint i1 = fetchIndex(idx, prim * 3 + 1);
	uint i2 = fetchIndex(idx, prim * 3 + 2);

	vec3 n0 = fetchNormal(verts, i0);
	vec3 n1 = fetchNormal(verts, i1);
	vec3 n2 = fetchNormal(verts, i2);
	vec3 p0 = fetchPos(verts, i0);
	vec3 p1 = fetchPos(verts, i1);
	vec3 p2 = fetchPos(verts, i2);

	vec3 gn = normalize(cross(p1 - p0, p2 - p0));
	vec3 n = normalize(n0 * (1.0 - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);
	if (dot(n, rayDir) > 0.0) {
		n = -n;
	}
	// Blend geometric + smooth normal for stability
	if (dot(gn, n) < 0.0) {
		gn = -gn;
	}
	n = normalize(mix(gn, n, 0.7));

	vec3 albedo = info.color.rgb;
	vec3 color = albedo * frame.ambient.rgb * frame.ambient.w;
	for (int i = 0; i < int(frame.rtParams.y); ++i) {
		SceneLight L = frame.lights[i];
		if (L.position.w == 0.0 && L.direction.w == 0.0) {
			color += albedo * L.color.rgb;
			continue;
		}
		vec3 lv = (L.direction.w == 1.0) ? -L.direction.xyz : L.position.xyz - (worldPos + rayDir * t);
		float ndl = max(dot(n, normalize(lv)), 0.0);
		color += albedo * L.color.rgb * ndl;
	}
	return color;
}

vec3 traceReflection(vec3 origin, vec3 dir, uint mask)
{
	rayQueryEXT rq;
	rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, mask, origin, 0.05, dir, 1e5);
	while (rayQueryProceedEXT(rq)) {
	}
	if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
		return shadeRayHit(rq, dir);
	}
	// Miss: sample the scene background color as a cheap sky.
	return frame.ambient.rgb;
}
#endif

float hash12(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distGGX(vec3 N, vec3 H, float rough)
{
	float a = rough * rough;
	float a2 = a * a;
	float ndh = max(dot(N, H), 0.0);
	float denom = ndh * ndh * (a2 - 1.0) + 1.0;
	return a2 / (3.14159265 * denom * denom);
}

float geomSmith(float ndv, float ndl, float rough)
{
	float r = rough + 1.0;
	float k = (r * r) / 8.0;
	float g1 = ndv / (ndv * (1.0 - k) + k);
	float g2 = ndl / (ndl * (1.0 - k) + k);
	return g1 * g2;
}

void main()
{
	vec3 N = normalize(worldNormal);
	vec3 V = normalize(frame.cameraPos.xyz - worldPos);
	if (dot(N, V) < 0.0) {
		N = -N;
	}

	vec4 base = draw.color;
	vec4 texel = vec4(1.0);
	if (draw.params.y != 0.0) {
		texel = texture(tex, uv);
		base *= texel;
	}

	// Unlit / flat surfaces pass color straight through.
	if (draw.params.z != 0.0) {
		outColor = vec4(base.rgb, base.a);
		return;
	}

	float shininess = draw.params.x;
	float rough = clamp(1.0 - shininess / 64.0, 0.03, 1.0);
	vec3 albedo = base.rgb;

	vec3 ambient = albedo * frame.ambient.rgb * frame.ambient.w;
	vec3 color = vec3(0.0);
	float ao = 1.0;

#ifdef RTX_RT
	uint rtFlags = uint(frame.rtParams.x + 0.5);
	// Ambient occlusion: a handful of short hemisphere rays.
	if ((rtFlags & 4u) != 0u) {
		float occ = 0.0;
		const int AO_RAYS = 4;
		for (int i = 0; i < AO_RAYS; ++i) {
			float h1 = hash12(gl_FragCoord.xy + vec2(i * 37.0, i * 11.0));
			float h2 = hash12(gl_FragCoord.yx + vec2(i * 17.0, i * 29.0));
			vec3 t1 = normalize(cross(N, abs(N.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
			vec3 t2 = cross(N, t1);
			float ang = h1 * 6.2831853;
			float rr = sqrt(h2);
			vec3 dir = normalize(t1 * cos(ang) * rr + t2 * sin(ang) * rr + N * sqrt(max(0.0, 1.0 - h2)));
			if (rayOccluded(worldPos + N * 0.2, dir, 0.0, 8.0, 0x1u)) {
				occ += 1.0;
			}
		}
		ao = 1.0 - 0.6 * (occ / float(AO_RAYS));
	}
#endif

	ambient *= ao;

	for (int i = 0; i < int(frame.rtParams.y); ++i) {
		SceneLight L = frame.lights[i];
		vec3 lc = L.color.rgb;

		if (L.position.w == 0.0 && L.direction.w == 0.0) {
			ambient += albedo * lc * ao;
			continue;
		}

		vec3 lightVec;
		float lightDist = 1e5;
		if (L.direction.w == 1.0) {
			lightVec = -L.direction.xyz;
		}
		else {
			lightVec = L.position.xyz - worldPos;
			lightDist = length(lightVec);
		}
		lightVec = normalize(lightVec);

		float ndl = max(dot(N, lightVec), 0.0);
		if (ndl <= 0.0) {
			continue;
		}

		float shadow = 1.0;
#ifdef RTX_RT
		if ((rtFlags & 1u) != 0u) {
			float tmax = (L.direction.w == 1.0) ? 1e4 : max(lightDist - 0.2, 0.0);
			if (rayOccluded(worldPos + N * 0.15, lightVec, 0.0, tmax, 0x1u)) {
				shadow = 0.0;
			}
		}
#endif

		// Cook-Torrance GGX specular + lambertian diffuse
		vec3 H = normalize(lightVec + V);
		float ndv = max(dot(N, V), 1e-4);
		float ndh = max(dot(N, H), 0.0);
		vec3 F0 = vec3(0.04);
		vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
		float D = distGGX(N, H, rough);
		float G = geomSmith(ndv, ndl, rough);
		vec3 spec = (D * G * F) / max(4.0 * ndv * ndl, 1e-4);
		vec3 kd = (vec3(1.0) - F);

		color += (kd * albedo + spec) * lc * ndl * shadow;
	}

	color += ambient;

#ifdef RTX_RT
	// Glossy reflections for shiny surfaces.
	if ((rtFlags & 2u) != 0u && rough < 0.45) {
		vec3 R = reflect(-V, N);
		float ndv = max(dot(N, V), 1e-4);
		vec3 F = fresnelSchlick(ndv, vec3(0.04));
		float w = clamp(F.x * 4.0 * (1.0 - rough), 0.0, 1.0);
		if (w > 0.003) {
			vec3 refl = traceReflection(worldPos + N * 0.2, R, 0x3u);
			color = mix(color, refl, w);
		}
	}
#endif

	outColor = vec4(color, base.a);
}
