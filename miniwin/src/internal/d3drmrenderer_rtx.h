#pragma once

#include "d3drmrenderer.h"
#include "d3drmtexture_impl.h"
#include "ddraw_impl.h"
#include "ddsurface_impl.h"

#include <SDL3/SDL.h>
#include <vector>
#include <vulkan/vulkan.h>

DEFINE_GUID(RTX_GUID, 0x52545831, 0x0001, 0x0001, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77);

// Mirrors FrameUBO in shaders/src/common.glsl (std140).
struct RTXFrameUBO {
	float proj[16];
	float view[16];
	float cameraPos[4];
	float ambient[4];
	float rtParams[4]; // x = rt flags, y = light count, z = front, w = back
	SceneLight lights[8];
};
static_assert(sizeof(RTXFrameUBO) == 64 + 64 + 16 + 16 + 16 + 48 * 8);

// Mirrors DrawUBO in shaders/src/common.glsl (std140). One 256-byte slot per draw.
struct RTXDrawUBO {
	float modelView[16];
	float normal[16];
	float world[16];
	float color[4];
	float params[4]; // x = shininess, y = useTexture, z = flat, w = unused
};
static_assert(sizeof(RTXDrawUBO) == 224);
#define RTX_DRAW_UBO_STRIDE 256

// Mirrors RTInstance in shaders/src/scene.frag.glsl (std430).
struct RTXInstanceInfo {
	uint32_t vbLo, vbHi;
	uint32_t ibLo, ibHi;
	float color[4];
	uint32_t indexCount;
	uint32_t flags;
	uint32_t textureId;
	uint32_t unused;
};
static_assert(sizeof(RTXInstanceInfo) == 48);

#define RTX_MAX_LIGHTS 8
#define RTX_MAX_DRAWS 8192
#define RTX_FRAMES_IN_FLIGHT 2

// RT effect flags stored in rtParams.x and ISLE_RTX_FLAGS.
#define RTX_EFFECT_SHADOWS 1u
#define RTX_EFFECT_REFLECTIONS 2u
#define RTX_EFFECT_AO 4u

struct RTXTextureCache {
	Direct3DRMTextureImpl* texture = nullptr;
	Uint32 version = 0;
	VkImage image = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
};

struct RTXMeshCache {
	const MeshGroup* meshGroup = nullptr;
	int version = 0;
	VkBuffer vertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
	VkBuffer indexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory indexMemory = VK_NULL_HANDLE;
	uint32_t indexCount = 0;
	uint32_t vertexCount = 0;
	VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
	VkBuffer blasBuffer = VK_NULL_HANDLE;
	VkDeviceMemory blasMemory = VK_NULL_HANDLE;
};

struct RTXDraw {
	uint32_t meshId;
	D3DRMMATRIX4D modelViewMatrix;
	D3DRMMATRIX4D worldMatrix;
	Matrix3x3 normalMatrix;
	Appearance appearance;
};

struct RTXUIDraw {
	uint32_t textureId;
	D3DRMMATRIX4D transform;
	SDL_Rect scissor;
	FColor color;
	uint32_t drawUBOSlot;
};

// Resources that must stay alive until a submitted command buffer completes.
struct RTXPendingOp {
	VkFence fence = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	std::vector<VkBuffer> buffers;
	std::vector<VkDeviceMemory> memories;
	std::vector<VkAccelerationStructureKHR> structures;
};

struct RTXFrameSlot {
	VkCommandPool cmdPool = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
	VkSemaphore renderDone = VK_NULL_HANDLE;
	VkBuffer frameUBO = VK_NULL_HANDLE;
	VkDeviceMemory frameUBOMemory = VK_NULL_HANDLE;
	void* frameUBOMap = nullptr;
	VkBuffer drawUBO = VK_NULL_HANDLE;
	VkDeviceMemory drawUBOMemory = VK_NULL_HANDLE;
	void* drawUBOMap = nullptr;
	VkBuffer instSSBO = VK_NULL_HANDLE;
	VkDeviceMemory instSSBOMemory = VK_NULL_HANDLE;
	void* instSSBOMap = nullptr;
	VkDeviceSize instSSBOSize = 0;
	VkBuffer tlasInstanceBuffer = VK_NULL_HANDLE;
	VkDeviceMemory tlasInstanceMemory = VK_NULL_HANDLE;
	VkDeviceSize tlasInstanceSize = 0;
	VkDescriptorSet set0 = VK_NULL_HANDLE;
	VkDescriptorPool drawDescPool = VK_NULL_HANDLE;
	// Resources referenced by this slot's own recorded commands.
	std::vector<VkBuffer> trashBuffers;
	std::vector<VkDeviceMemory> trashMemories;
	bool submitted = false;
};

class Direct3DRMRTXRenderer : public Direct3DRMRenderer {
public:
	static Direct3DRMRenderer* Create(DWORD width, DWORD height);
	~Direct3DRMRTXRenderer() override;
	void PushLights(const SceneLight* vertices, size_t count) override;
	Uint32 GetTextureId(IDirect3DRMTexture* texture, bool isUI, float scaleX, float scaleY) override;
	Uint32 GetMeshId(IDirect3DRMMesh* mesh, const MeshGroup* meshGroup) override;
	void SetProjection(const D3DRMMATRIX4D& projection, D3DVALUE front, D3DVALUE back) override;
	void SetFrustumPlanes(const Plane* frustumPlanes) override;
	HRESULT BeginFrame() override;
	void EnableTransparency() override;
	void SubmitDraw(
		DWORD meshId,
		const D3DRMMATRIX4D& modelViewMatrix,
		const D3DRMMATRIX4D& worldMatrix,
		const D3DRMMATRIX4D& viewMatrix,
		const Matrix3x3& normalMatrix,
		const Appearance& appearance
	) override;
	HRESULT FinalizeFrame() override;
	void Resize(int width, int height, const ViewportTransform& viewportTransform) override;
	void Clear(float r, float g, float b) override;
	void Flip() override;
	void Draw2DImage(Uint32 textureId, const SDL_Rect& srcRect, const SDL_Rect& dstRect, FColor color) override;
	void Download(SDL_Surface* target) override;
	void SetDither(bool dither) override;

private:
	Direct3DRMRTXRenderer(DWORD width, DWORD height);

	bool InitInstance();
	bool InitDevice();
	bool InitSwapchain();
	bool InitTargets();
	bool InitPipelines();
	bool InitFrameSlots();
	void DestroySwapchain();
	void DestroyTargets();
	void DestroyDeviceResources();

	uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);
	bool CreateBuffer(
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags props,
		VkBuffer& buffer,
		VkDeviceMemory& memory,
		bool hostMapped = false,
		void** mapped = nullptr
	);
	bool CreateImage2D(
		uint32_t w,
		uint32_t h,
		VkFormat format,
		VkImageUsageFlags usage,
		VkImage& image,
		VkDeviceMemory& memory,
		VkImageView& view
	);
	VkShaderModule CreateShaderModule(const uint32_t* code, size_t size);
	VkDeviceAddress BufferAddress(VkBuffer buffer);
	VkCommandBuffer BeginImmediate();
	// Submits an immediate command buffer and tracks its resources until done.
	void SubmitImmediate(VkCommandBuffer cmd, std::vector<VkBuffer> bufs, std::vector<VkDeviceMemory> mems);
	void ReapPending();
	void RetireBuffer(VkBuffer b, VkDeviceMemory m);
	void RetireAS(VkAccelerationStructureKHR as, VkBuffer b, VkDeviceMemory m);

	bool CreateBLAS(RTXMeshCache& mesh);
	bool BuildTLAS(RTXFrameSlot& slot, VkCommandBuffer cmd);
	bool CreateDummyTLAS();
	RTXMeshCache UploadMeshData(const MeshGroup& meshGroup, bool buildBLAS);
	bool UploadTexture(RTXTextureCache& entry, SDL_Surface* surface);

	void WaitForSlot(RTXFrameSlot& slot);
	bool RecordFrame();
	void WriteFrameUBO(RTXFrameSlot& slot);
	VkDescriptorSet AllocDrawSet(RTXFrameSlot& slot, uint32_t uboOffset, VkImageView texView, VkSampler sampler);
	void UpdateSet0(RTXFrameSlot& slot);

	VkPipeline CreateScenePipeline(const uint32_t* fragSpv, size_t fragSize, bool blend, bool depthWrite);
	VkPipeline CreateUIPipeline();
	VkPipeline CreatePostPipeline();

	VkInstance m_instance = VK_NULL_HANDLE;
	VkSurfaceKHR m_surface = VK_NULL_HANDLE;
	VkPhysicalDevice m_phys = VK_NULL_HANDLE;
	VkDevice m_device = VK_NULL_HANDLE;
	VkQueue m_queue = VK_NULL_HANDLE;
	uint32_t m_queueFamily = 0;
	bool m_rtSupported = false;
	bool m_rtEnabled = false;
	uint32_t m_rtFlags = 0;
	char m_deviceName[256] = {};

	PFN_vkCreateAccelerationStructureKHR pfnCreateAS = nullptr;
	PFN_vkDestroyAccelerationStructureKHR pfnDestroyAS = nullptr;
	PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetASBuildSizes = nullptr;
	PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAS = nullptr;
	PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASAddr = nullptr;
	PFN_vkGetBufferDeviceAddress pfnGetBufferAddress = nullptr;

	VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
	std::vector<VkImage> m_swapImages;
	std::vector<VkImageView> m_swapViews;
	VkFormat m_swapFormat = VK_FORMAT_B8G8R8A8_UNORM;
	VkExtent2D m_swapExtent = {0, 0};
	uint32_t m_presentImage = 0;
	bool m_presentPending = false;
	VkSemaphore m_acquireSem = VK_NULL_HANDLE;
	VkCommandPool m_immediatePool = VK_NULL_HANDLE;
	std::vector<RTXPendingOp> m_pendingOps;

	VkImage m_hdrImage = VK_NULL_HANDLE;
	VkDeviceMemory m_hdrMemory = VK_NULL_HANDLE;
	VkImageView m_hdrView = VK_NULL_HANDLE;
	VkImage m_ldrImage = VK_NULL_HANDLE;
	VkDeviceMemory m_ldrMemory = VK_NULL_HANDLE;
	VkImageView m_ldrView = VK_NULL_HANDLE;
	VkImage m_depthImage = VK_NULL_HANDLE;
	VkDeviceMemory m_depthMemory = VK_NULL_HANDLE;
	VkImageView m_depthView = VK_NULL_HANDLE;

	VkRenderPass m_hdrPass = VK_NULL_HANDLE;
	VkRenderPass m_ldrPass = VK_NULL_HANDLE;
	VkFramebuffer m_hdrFB = VK_NULL_HANDLE;
	VkFramebuffer m_ldrFB = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_set0RT = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_set0Plain = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_set1 = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_postSet0 = VK_NULL_HANDLE;
	VkDescriptorPool m_set0Pool = VK_NULL_HANDLE;
	VkDescriptorSet m_postSet = VK_NULL_HANDLE;
	VkPipelineLayout m_sceneLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_postLayout = VK_NULL_HANDLE;
	VkPipeline m_opaquePipeline = VK_NULL_HANDLE;
	VkPipeline m_transparentPipeline = VK_NULL_HANDLE;
	VkPipeline m_uiPipeline = VK_NULL_HANDLE;
	VkPipeline m_postPipeline = VK_NULL_HANDLE;
	VkSampler m_sampler = VK_NULL_HANDLE;
	VkSampler m_uiSampler = VK_NULL_HANDLE;

	VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
	VkBuffer m_tlasBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_tlasMemory = VK_NULL_HANDLE;
	VkDeviceSize m_tlasSize = 0;
	VkAccelerationStructureKHR m_dummyBLAS = VK_NULL_HANDLE;
	VkBuffer m_dummyBLASBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_dummyBLASMemory = VK_NULL_HANDLE;
	std::vector<VkBuffer> m_keepAliveBuffers;
	std::vector<VkDeviceMemory> m_keepAliveMemories;

	RTXFrameSlot m_slots[RTX_FRAMES_IN_FLIGHT];
	uint32_t m_slot = 0;

	MeshGroup m_uiMesh;
	RTXMeshCache m_uiMeshCache;

	std::vector<RTXTextureCache> m_textures;
	std::vector<RTXMeshCache> m_meshs;
	std::vector<RTXDraw> m_draws;
	std::vector<RTXDraw> m_transparentDraws;
	std::vector<RTXUIDraw> m_uiDraws;
	std::vector<SceneLight> m_lights;
	D3DRMMATRIX4D m_projection = {};
	D3DRMMATRIX4D m_viewMatrix = {};
	bool m_gotView = false;
	float m_front = 0.f;
	float m_back = 0.f;
	float m_clearColor[4] = {0.f, 0.f, 0.f, 1.f};
	bool m_frameDirty = false;
	bool m_transparencyEnabled = false;
	bool m_dither = true;
	VkImage m_dummyImage = VK_NULL_HANDLE;
	VkDeviceMemory m_dummyMemory = VK_NULL_HANDLE;
	VkImageView m_dummyView = VK_NULL_HANDLE;
};

bool Direct3DRMRTX_Available();

inline static void Direct3DRMRTX_EnumDevice(LPD3DENUMDEVICESCALLBACK cb, void* ctx)
{
	if (!Direct3DRMRTX_Available()) {
		return;
	}
	D3DDEVICEDESC halDesc = {};
	halDesc.dcmColorModel = D3DCOLOR_RGB;
	halDesc.dwFlags = D3DDD_DEVICEZBUFFERBITDEPTH;
	halDesc.dwDeviceZBufferBitDepth = DDBD_32;
	halDesc.dwDeviceRenderBitDepth = DDBD_32;
	halDesc.dpcTriCaps.dwTextureCaps = D3DPTEXTURECAPS_PERSPECTIVE;
	halDesc.dpcTriCaps.dwShadeCaps = D3DPSHADECAPS_ALPHAFLATBLEND;
	halDesc.dpcTriCaps.dwTextureFilterCaps = D3DPTFILTERCAPS_LINEAR;

	D3DDEVICEDESC helDesc = {};

	EnumDevice(cb, ctx, "LEGO Island RTX (Vulkan)", &halDesc, &helDesc, RTX_GUID);
}
