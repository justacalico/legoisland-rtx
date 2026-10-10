// Headless-capable smoke test for the Vulkan RTX D3DRM backend.
// Renders a test scene (floor + rotating cube + UI quad) through the real
// renderer interface, downloads the framebuffer and checks pixel statistics.

#include "d3drmrenderer_rtx.h"
#include "d3drmmesh_impl.h"
#include "d3drmtexture_impl.h"
#include "ddraw_impl.h"
#include "ddsurface_impl.h"
#include "meshutils.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <cmath>
#include <cstdio>
#include <vector>

static void MatMul(D3DRMMATRIX4D& out, const D3DRMMATRIX4D& a, const D3DRMMATRIX4D& b)
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			out[i][j] = 0;
			for (int k = 0; k < 4; ++k) {
				out[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

// D3DRM-style left-handed look-at view matrix (row-vector convention).
static void LookAtLH(D3DRMMATRIX4D& out, float ex, float ey, float ez, float tx, float ty, float tz)
{
	float fx = tx - ex, fy = ty - ey, fz = tz - ez;
	float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
	fx /= fl;
	fy /= fl;
	fz /= fl;
	float ux = 0, uy = 1, uz = 0;
	// right = normalize(cross(up, fwd))
	float rx = uy * fz - uz * fy;
	float ry = uz * fx - ux * fz;
	float rz = ux * fy - uy * fx;
	float rl = std::sqrt(rx * rx + ry * ry + rz * rz);
	rx /= rl;
	ry /= rl;
	rz /= rl;
	// up' = cross(fwd, right)
	float ux2 = fy * rz - fz * ry;
	float uy2 = fz * rx - fx * rz;
	float uz2 = fx * ry - fy * rx;

	D3DRMMATRIX4D m = {
		{rx, ux2, fx, 0},
		{ry, uy2, fy, 0},
		{rz, uz2, fz, 0},
		{-(rx * ex + ry * ey + rz * ez), -(ux2 * ex + uy2 * ey + uz2 * ez), -(fx * ex + fy * ey + fz * ez), 1}
	};
	memcpy(out, m, sizeof(m));
}

static Direct3DRMMeshImpl* MakeMesh(const std::vector<D3DRMVERTEX>& verts, const std::vector<DWORD>& indices)
{
	auto* mesh = new Direct3DRMMeshImpl();
	std::vector<unsigned int> faces(indices.begin(), indices.end());
	D3DRMGROUPINDEX groupIndex = 0;
	mesh->AddGroup((unsigned) verts.size(), (unsigned) (indices.size() / 3), 3, faces.data(), &groupIndex);
	std::vector<D3DRMVERTEX> mutableVerts = verts;
	mesh->SetVertices(groupIndex, 0, (int) verts.size(), mutableVerts.data());
	return mesh;
}

static void MakeCube(std::vector<D3DRMVERTEX>& verts, std::vector<DWORD>& indices, float s)
{
	// 6 faces, outward normals
	const float n[6][3] = {
		{0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}
	};
	const int faces[6][4] = {
		{0, 2, 3, 1}, // -z
		{4, 5, 7, 6}, // +z
		{0, 1, 5, 4}, // -x
		{2, 6, 7, 3}, // +x
		{0, 4, 6, 2}, // -y
		{1, 3, 7, 5} // +y
	};
	const float p[8][3] = {
		{-s, -s, -s}, {s, -s, -s}, {-s, s, -s}, {s, s, -s},
		{-s, -s, s},  {s, -s, s},  {-s, s, s},  {s, s, s}
	};
	for (int f = 0; f < 6; ++f) {
		unsigned base = (unsigned) verts.size();
		for (int v = 0; v < 4; ++v) {
			D3DRMVERTEX dv = {};
			dv.position = {p[faces[f][v]][0], p[faces[f][v]][1], p[faces[f][v]][2]};
			dv.normal = {n[f][0], n[f][1], n[f][2]};
			dv.texCoord = {(float) (v == 1 || v == 2), (float) (v >= 2)};
			verts.push_back(dv);
		}
		indices.push_back(base + 0);
		indices.push_back(base + 1);
		indices.push_back(base + 2);
		indices.push_back(base + 0);
		indices.push_back(base + 2);
		indices.push_back(base + 3);
	}
}

int main(int argc, char** argv)
{
	int outPpm = 0;
	for (int i = 1; i < argc; ++i) {
		if (SDL_strcmp(argv[i], "--ppm") == 0 && i + 1 < argc) {
			outPpm = i + 1;
		}
	}

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		SDL_Log("SDL_Init failed: %s", SDL_GetError());
		return 1;
	}
	SDL_Window* window =
		SDL_CreateWindow("rtxtest", 960, 540, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
	if (!window) {
		SDL_Log("No window; running surface-less");
	}
	DDWindow = window;

	auto* renderer = Direct3DRMRTXRenderer::Create(960, 540);
	if (!renderer) {
		SDL_Log("FAIL: could not create RTX renderer");
		return 2;
	}
	SDL_Log("RTX renderer created");

	// Floor: large quad, shiny (shininess drives gloss + reflections).
	std::vector<D3DRMVERTEX> fv;
	std::vector<DWORD> fi;
	{
		float s = 200.f;
		D3DRMVERTEX v[4] = {
			{{-s, 0, -s}, {0, 1, 0}, {0, 0}},
			{{s, 0, -s}, {0, 1, 0}, {1, 0}},
			{{s, 0, s}, {0, 1, 0}, {1, 1}},
			{{-s, 0, s}, {0, 1, 0}, {0, 1}},
		};
		fv.assign(v, v + 4);
		fi = {0, 2, 1, 0, 3, 2};
	}
	auto* floorMesh = MakeMesh(fv, fi);
	auto& floorGroup = const_cast<MeshGroup&>(floorMesh->GetGroup(0));
	floorGroup.color = {200, 200, 210, 255};

	std::vector<D3DRMVERTEX> cv;
	std::vector<DWORD> ci;
	MakeCube(cv, ci, 15.f);
	auto* cubeMesh = MakeMesh(cv, ci);
	auto& cubeGroup = const_cast<MeshGroup&>(cubeMesh->GetGroup(0));
	cubeGroup.color = {230, 60, 60, 255};

	// UI texture: solid green 8x8.
	auto* ddsurface = new DirectDrawSurfaceImpl(8, 8, SDL_PIXELFORMAT_RGBA32);
	SDL_FillSurfaceRect(ddsurface->m_surface, nullptr, 0xFF00FF00);
	auto* texObj = new Direct3DRMTextureImpl(ddsurface, false);
	Uint32 uiTexId = renderer->GetTextureId(texObj);

	Uint32 floorMeshId = renderer->GetMeshId(floorMesh, &floorGroup);
	Uint32 cubeMeshId = renderer->GetMeshId(cubeMesh, &cubeGroup);

	SceneLight lights[2] = {};
	lights[0].color = {1.0f, 0.95f, 0.85f, 1.0f};
	lights[0].direction = {0.9f, -1.0f, 0.5f};
	lights[0].directional = 1.0f;
	lights[0].position = {0, 0, 0};
	lights[0].positional = 0.0f;
	lights[1].color = {0.2f, 0.25f, 0.35f, 1.0f}; // ambient
	lights[1].position = {0, 0, 0};
	lights[1].direction = {0, 0, 0};

	D3DRMMATRIX4D view;
	LookAtLH(view, -40, 80, -140, 0, 0, 20);
	D3DRMMATRIX4D proj = {
		{1.5f, 0, 0, 0},
		{0, 2.0f, 0, 0},
		{0, 0, 1000.0f / 999.0f, 1},
		{0, 0, -1000.0f / 999.0f, 0}
	};
	renderer->SetProjection(proj, 1.0f, 1000.0f);

	D3DRMMATRIX4D identity = {
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1}
	};
	Matrix3x3 identNormal = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	Appearance floorApp = {{200, 200, 210, 255}, 48.0f, NO_TEXTURE_ID, 0};
	Appearance cubeApp = {{230, 60, 60, 255}, 8.0f, NO_TEXTURE_ID, 0};

	const int frames = 60;
	for (int f = 0; f < frames; ++f) {
		float ang = f * 0.05f;
		float c = std::cos(ang), s = std::sin(ang);
		D3DRMMATRIX4D cubeWorld = {
			{c, 0, -s, 0},
			{0, 1, 0, 0},
			{s, 0, c, 0},
			{20 * s, 20, 0, 1}
		};
		D3DRMMATRIX4D mv;
		renderer->Clear(0.02f, 0.03f, 0.08f);
		renderer->PushLights(lights, 2);
		renderer->BeginFrame();
		MatMul(mv, identity, view);
		renderer->SubmitDraw(floorMeshId, mv, identity, view, identNormal, floorApp);
		MatMul(mv, cubeWorld, view);
		renderer->SubmitDraw(cubeMeshId, mv, cubeWorld, view, identNormal, cubeApp);
		renderer->EnableTransparency();
		renderer->FinalizeFrame();
		SDL_Rect src = {0, 0, 8, 8};
		SDL_Rect dst = {20, 20, 64, 64};
		FColor uiColor = {1, 1, 1, 1};
		renderer->Draw2DImage(uiTexId, src, dst, uiColor);
		renderer->Flip();
		SDL_PumpEvents();
		SDL_Delay(8);
	}

	// Read back and analyze.
	SDL_Surface* shot = SDL_CreateSurface(960, 540, SDL_PIXELFORMAT_RGBA32);
	renderer->Download(shot);
	if (!shot->pixels) {
		SDL_Log("FAIL: Download returned no pixels");
		return 3;
	}
	Uint32* px = (Uint32*) shot->pixels;
	auto at = [&](int x, int y) { return px[y * (shot->pitch / 4) + x]; };

	long nonBg = 0;
	double rSum = 0, gSum = 0, bSum = 0;
	for (int y = 0; y < shot->h; ++y) {
		for (int x = 0; x < shot->w; ++x) {
			Uint32 p = at(x, y);
			Uint8 r = p & 0xFF, g = (p >> 8) & 0xFF, b = (p >> 16) & 0xFF;
			rSum += r;
			gSum += g;
			bSum += b;
			Uint8 br = (Uint8) (0.02 * 255), bgc = (Uint8) (0.03 * 255), bb = (Uint8) (0.08 * 255);
			if (std::abs((int) r - br) > 12 || std::abs((int) g - bgc) > 12 || std::abs((int) b - bb) > 16) {
				nonBg++;
			}
		}
	}
	Uint32 center = at(480, 270);
	Uint32 uiPx = at(50, 50);
	SDL_Log("center pixel: %08x", center);
	SDL_Log("ui pixel: %08x", uiPx);
	SDL_Log("non-background pixels: %ld", nonBg);

	if (outPpm) {
		FILE* f = fopen(argv[outPpm], "wb");
		if (f) {
			fprintf(f, "P6\n%d %d\n255\n", shot->w, shot->h);
			for (int y = 0; y < shot->h; ++y) {
				for (int x = 0; x < shot->w; ++x) {
					Uint32 p = at(x, y);
					Uint8 rgb[3] = {(Uint8) (p & 0xFF), (Uint8) ((p >> 8) & 0xFF), (Uint8) ((p >> 16) & 0xFF)};
					fwrite(rgb, 1, 3, f);
				}
			}
			fclose(f);
			SDL_Log("wrote %s", argv[outPpm]);
		}
	}

	int failures = 0;
	if (nonBg < 20000) {
		SDL_Log("FAIL: scene appears empty (%ld non-bg pixels)", nonBg);
		failures++;
	}
	// Cube is red: center region should contain some red-ish pixels.
	{
		int redPx = 0;
		for (int y = 200; y < 340; ++y) {
			for (int x = 400; x < 560; ++x) {
				Uint32 p = at(x, y);
				if ((p & 0xFF) > 120 && ((p >> 8) & 0xFF) < 110) {
					redPx++;
				}
			}
		}
		if (redPx < 50) {
			SDL_Log("FAIL: red cube not found in center region (%d px)", redPx);
			failures++;
		}
	}
	// UI quad should be green.
	{
		Uint8 r = uiPx & 0xFF, g = (uiPx >> 8) & 0xFF, b = (uiPx >> 16) & 0xFF;
		if (!(g > 120 && r < 100 && b < 100)) {
			SDL_Log("FAIL: UI quad not green (r=%u g=%u b=%u)", r, g, b);
			failures++;
		}
	}

	texObj->Release();
	ddsurface->Release();
	cubeMesh->Release();
	floorMesh->Release();
	renderer->Release();

	SDL_DestroyWindow(window);
	SDL_Quit();

	if (failures) {
		SDL_Log("RESULT: %d failure(s)", failures);
		return 4;
	}
	SDL_Log("RESULT: all checks passed");
	return 0;
}
