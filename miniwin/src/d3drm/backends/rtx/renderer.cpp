// LEGO Island RTX renderer: a Vulkan backend for miniwin's D3DRM emulation.
//
// - HDR pipeline (R16G16B16A16_SFLOAT) with Cook-Torrance GGX PBR shading
// - ACES filmic tonemap + bloom + dither post pass
// - Hardware ray tracing via VK_KHR_acceleration_structure + VK_KHR_ray_query:
//   ray-traced shadows, ambient occlusion and glossy reflections.
//   Works on NVIDIA RTX and AMD RDNA2+ on Windows and Linux. On devices
//   without ray query support (e.g. MoltenVK on macOS) the renderer still
//   provides the full PBR raster path.

#include "d3drmrenderer_rtx.h"

#include "ddraw_impl.h"
#include "mathutils.h"
#include "meshutils.h"
#include "miniwin.h"

#include "shaders/generated/post.frag.h"
#include "shaders/generated/post.vert.h"
#include "shaders/generated/scene.frag.h"
#include "shaders/generated/scene.vert.h"
#include "shaders/generated/scene_rt.frag.h"
#include "shaders/generated/ui.frag.h"
#include "shaders/generated/ui.vert.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#define VK_CHECK(x)                                                                                 \
	do {                                                                                            \
		VkResult _res = (x);                                                                        \
		if (_res != VK_SUCCESS) {                                                                   \
			SDL_LogError(LOG_CATEGORY_MINIWIN, "%s failed: %d", #x, (int) _res);                     \
			return false;                                                                           \
		}                                                                                           \
	} while (0)

bool Direct3DRMRTX_Available()
{
	if (!SDL_Vulkan_LoadLibrary(nullptr)) {
		return false;
	}
	Uint32 extCount = 0;
	const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&extCount);
	VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ci.pApplicationInfo = &app;
	ci.enabledExtensionCount = extCount;
	ci.ppEnabledExtensionNames = sdlExts;
	VkInstance inst;
	if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
		SDL_Vulkan_UnloadLibrary();
		return false;
	}
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(inst, &count, nullptr);
	vkDestroyInstance(inst, nullptr);
	SDL_Vulkan_UnloadLibrary();
	return count > 0;
}

Direct3DRMRenderer* Direct3DRMRTXRenderer::Create(DWORD width, DWORD height)
{
	auto* renderer = new Direct3DRMRTXRenderer(width, height);
	if (!renderer->InitInstance() || !renderer->InitDevice() || !renderer->InitSwapchain() ||
		!renderer->InitTargets() || !renderer->InitPipelines() || !renderer->InitFrameSlots()) {
		delete renderer;
		return nullptr;
	}
	return renderer;
}

Direct3DRMRTXRenderer::Direct3DRMRTXRenderer(DWORD width, DWORD height)
{
	m_width = width;
	m_height = height;
	m_virtualWidth = width;
	m_virtualHeight = height;
	m_viewportTransform = {1.0f, 0.0f, 0.0f};
	m_rtFlags = RTX_EFFECT_SHADOWS | RTX_EFFECT_REFLECTIONS | RTX_EFFECT_AO;
	if (const char* flags = SDL_getenv("ISLE_RTX_FLAGS")) {
		m_rtFlags = (uint32_t) SDL_atoi(flags);
	}
	if (const char* en = SDL_getenv("ISLE_RTX")) {
		if (SDL_atoi(en) == 0) {
			m_rtFlags = 0;
		}
	}
}

Direct3DRMRTXRenderer::~Direct3DRMRTXRenderer()
{
	assert(m_refCount == 0);
	if (m_device) {
		vkDeviceWaitIdle(m_device);
	}
	for (auto& op : m_pendingOps) {
		if (op.fence) {
			vkDestroyFence(m_device, op.fence, nullptr);
		}
		if (op.cmd) {
			vkFreeCommandBuffers(m_device, m_immediatePool, 1, &op.cmd);
		}
		for (auto b : op.buffers) {
			vkDestroyBuffer(m_device, b, nullptr);
		}
		for (auto m : op.memories) {
			vkFreeMemory(m_device, m, nullptr);
		}
		for (auto as : op.structures) {
			if (pfnDestroyAS) {
				pfnDestroyAS(m_device, as, nullptr);
			}
		}
	}
	m_pendingOps.clear();
	for (auto& slot : m_slots) {
		if (slot.cmdPool) {
			vkDestroyCommandPool(m_device, slot.cmdPool, nullptr);
		}
		if (slot.fence) {
			vkDestroyFence(m_device, slot.fence, nullptr);
		}
		if (slot.renderDone) {
			vkDestroySemaphore(m_device, slot.renderDone, nullptr);
		}
		if (slot.frameUBO) {
			vkDestroyBuffer(m_device, slot.frameUBO, nullptr);
			vkFreeMemory(m_device, slot.frameUBOMemory, nullptr);
		}
		if (slot.drawUBO) {
			vkDestroyBuffer(m_device, slot.drawUBO, nullptr);
			vkFreeMemory(m_device, slot.drawUBOMemory, nullptr);
		}
		if (slot.instSSBO) {
			vkDestroyBuffer(m_device, slot.instSSBO, nullptr);
			vkFreeMemory(m_device, slot.instSSBOMemory, nullptr);
		}
		if (slot.tlasInstanceBuffer) {
			vkDestroyBuffer(m_device, slot.tlasInstanceBuffer, nullptr);
			vkFreeMemory(m_device, slot.tlasInstanceMemory, nullptr);
		}
		if (slot.drawDescPool) {
			vkDestroyDescriptorPool(m_device, slot.drawDescPool, nullptr);
		}
		for (auto b : slot.trashBuffers) {
			vkDestroyBuffer(m_device, b, nullptr);
		}
		for (auto m : slot.trashMemories) {
			vkFreeMemory(m_device, m, nullptr);
		}
	}
	for (auto& mesh : m_meshs) {
		if (m_device) {
			vkDestroyBuffer(m_device, mesh.vertexBuffer, nullptr);
			vkFreeMemory(m_device, mesh.vertexMemory, nullptr);
			vkDestroyBuffer(m_device, mesh.indexBuffer, nullptr);
			vkFreeMemory(m_device, mesh.indexMemory, nullptr);
			if (mesh.blas && pfnDestroyAS) {
				pfnDestroyAS(m_device, mesh.blas, nullptr);
				vkDestroyBuffer(m_device, mesh.blasBuffer, nullptr);
				vkFreeMemory(m_device, mesh.blasMemory, nullptr);
			}
		}
	}
	for (auto& tex : m_textures) {
		if (tex.image) {
			vkDestroyImageView(m_device, tex.view, nullptr);
			vkDestroyImage(m_device, tex.image, nullptr);
			vkFreeMemory(m_device, tex.memory, nullptr);
		}
	}
	if (m_uiMeshCache.vertexBuffer) {
		vkDestroyBuffer(m_device, m_uiMeshCache.vertexBuffer, nullptr);
		vkFreeMemory(m_device, m_uiMeshCache.vertexMemory, nullptr);
		vkDestroyBuffer(m_device, m_uiMeshCache.indexBuffer, nullptr);
		vkFreeMemory(m_device, m_uiMeshCache.indexMemory, nullptr);
	}
	if (m_tlas && pfnDestroyAS) {
		pfnDestroyAS(m_device, m_tlas, nullptr);
		vkDestroyBuffer(m_device, m_tlasBuffer, nullptr);
		vkFreeMemory(m_device, m_tlasMemory, nullptr);
	}
	if (m_dummyBLAS && pfnDestroyAS) {
		pfnDestroyAS(m_device, m_dummyBLAS, nullptr);
		vkDestroyBuffer(m_device, m_dummyBLASBuffer, nullptr);
		vkFreeMemory(m_device, m_dummyBLASMemory, nullptr);
	}
	for (auto b : m_keepAliveBuffers) {
		vkDestroyBuffer(m_device, b, nullptr);
	}
	for (auto m : m_keepAliveMemories) {
		vkFreeMemory(m_device, m, nullptr);
	}
	if (m_dummyImage) {
		vkDestroyImageView(m_device, m_dummyView, nullptr);
		vkDestroyImage(m_device, m_dummyImage, nullptr);
		vkFreeMemory(m_device, m_dummyMemory, nullptr);
	}
	if (m_device) {
		DestroyTargets();
		DestroySwapchain();
		if (m_opaquePipeline) {
			vkDestroyPipeline(m_device, m_opaquePipeline, nullptr);
		}
		if (m_transparentPipeline) {
			vkDestroyPipeline(m_device, m_transparentPipeline, nullptr);
		}
		if (m_uiPipeline) {
			vkDestroyPipeline(m_device, m_uiPipeline, nullptr);
		}
		if (m_postPipeline) {
			vkDestroyPipeline(m_device, m_postPipeline, nullptr);
		}
		if (m_sceneLayout) {
			vkDestroyPipelineLayout(m_device, m_sceneLayout, nullptr);
		}
		if (m_postLayout) {
			vkDestroyPipelineLayout(m_device, m_postLayout, nullptr);
		}
		if (m_set0RT) {
			vkDestroyDescriptorSetLayout(m_device, m_set0RT, nullptr);
			vkDestroyDescriptorSetLayout(m_device, m_set0Plain, nullptr);
			vkDestroyDescriptorSetLayout(m_device, m_set1, nullptr);
			vkDestroyDescriptorSetLayout(m_device, m_postSet0, nullptr);
		}
		if (m_set0Pool) {
			vkDestroyDescriptorPool(m_device, m_set0Pool, nullptr);
		}
		if (m_sampler) {
			vkDestroySampler(m_device, m_sampler, nullptr);
			vkDestroySampler(m_device, m_uiSampler, nullptr);
		}
		if (m_acquireSem) {
			vkDestroySemaphore(m_device, m_acquireSem, nullptr);
		}
		if (m_immediatePool) {
			vkDestroyCommandPool(m_device, m_immediatePool, nullptr);
		}
		vkDestroyDevice(m_device, nullptr);
	}
	if (m_instance) {
		if (m_surface) {
			SDL_Vulkan_DestroySurface(m_instance, m_surface, nullptr);
		}
		vkDestroyInstance(m_instance, nullptr);
	}
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool Direct3DRMRTXRenderer::InitInstance()
{
	if (!SDL_Vulkan_LoadLibrary(nullptr)) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "SDL_Vulkan_LoadLibrary: %s", SDL_GetError());
		return false;
	}
	Uint32 extCount = 0;
	const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&extCount);

	uint32_t availExtCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &availExtCount, nullptr);
	std::vector<VkExtensionProperties> availExts(availExtCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &availExtCount, availExts.data());

	std::vector<const char*> exts;
	if (sdlExts) {
		exts.assign(sdlExts, sdlExts + extCount);
	}
	bool portability = false;
	for (auto& e : availExts) {
		if (SDL_strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
			portability = true;
		}
	}

	VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "LEGO Island RTX";
	app.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ci.pApplicationInfo = &app;
	if (portability) {
		exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}
	ci.enabledExtensionCount = (uint32_t) exts.size();
	ci.ppEnabledExtensionNames = exts.data();

	if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "vkCreateInstance failed");
		return false;
	}

	if (DDWindow) {
		if (!SDL_Vulkan_CreateSurface(DDWindow, m_instance, nullptr, &m_surface)) {
			SDL_LogWarn(
				LOG_CATEGORY_MINIWIN,
				"SDL_Vulkan_CreateSurface failed (%s), running headless",
				SDL_GetError()
			);
			m_surface = VK_NULL_HANDLE;
		}
	}
	return true;
}

static bool RTXHasExt(const std::vector<VkExtensionProperties>& props, const char* name)
{
	for (auto& e : props) {
		if (SDL_strcmp(e.extensionName, name) == 0) {
			return true;
		}
	}
	return false;
}

bool Direct3DRMRTXRenderer::InitDevice()
{
	uint32_t physCount = 0;
	vkEnumeratePhysicalDevices(m_instance, &physCount, nullptr);
	if (!physCount) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "No Vulkan physical devices");
		return false;
	}
	std::vector<VkPhysicalDevice> phys(physCount);
	vkEnumeratePhysicalDevices(m_instance, &physCount, phys.data());

	const char* wantDevice = SDL_getenv("ISLE_RTX_DEVICE");

	VkPhysicalDevice best = VK_NULL_HANDLE;
	uint32_t bestQueue = 0;
	bool bestRT = false;
	int bestScore = -1;

	for (VkPhysicalDevice p : phys) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(p, &props);
		if (wantDevice && !SDL_strstr(props.deviceName, wantDevice)) {
			continue;
		}

		uint32_t qCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(p, &qCount, nullptr);
		std::vector<VkQueueFamilyProperties> qProps(qCount);
		vkGetPhysicalDeviceQueueFamilyProperties(p, &qCount, qProps.data());
		uint32_t qFam = UINT32_MAX;
		for (uint32_t i = 0; i < qCount; ++i) {
			if (!(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				continue;
			}
			if (m_surface) {
				VkBool32 present = VK_FALSE;
				vkGetPhysicalDeviceSurfaceSupportKHR(p, i, m_surface, &present);
				if (!present) {
					continue;
				}
			}
			qFam = i;
			break;
		}
		if (qFam == UINT32_MAX) {
			continue;
		}

		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionProperties(p, nullptr, &extCount, nullptr);
		std::vector<VkExtensionProperties> extProps(extCount);
		vkEnumerateDeviceExtensionProperties(p, nullptr, &extCount, extProps.data());
		if (m_surface && !RTXHasExt(extProps, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
			continue;
		}

		bool rt = false;
		if (RTXHasExt(extProps, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
			RTXHasExt(extProps, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
			RTXHasExt(extProps, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
			VkPhysicalDeviceRayQueryFeaturesKHR rqFeat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
			VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat = {
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rqFeat
			};
			VkPhysicalDeviceVulkan12Features v12 = {
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &asFeat
			};
			VkPhysicalDeviceFeatures2 feats = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
			vkGetPhysicalDeviceFeatures2(p, &feats);
			rt = rqFeat.rayQuery && asFeat.accelerationStructure && v12.bufferDeviceAddress;
		}

		int score = (rt ? 4000 : 0) +
					(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 0) +
					(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 500 : 0);
		if (score > bestScore) {
			bestScore = score;
			best = p;
			bestQueue = qFam;
			bestRT = rt;
		}
	}

	if (!best) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "No suitable Vulkan device found");
		return false;
	}
	m_phys = best;
	m_queueFamily = bestQueue;
	m_rtSupported = bestRT;
	m_rtEnabled = bestRT && m_rtFlags != 0;

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_phys, &props);
	SDL_strlcpy(m_deviceName, props.deviceName, sizeof(m_deviceName));

	uint32_t extCount = 0;
	vkEnumerateDeviceExtensionProperties(m_phys, nullptr, &extCount, nullptr);
	std::vector<VkExtensionProperties> extProps(extCount);
	vkEnumerateDeviceExtensionProperties(m_phys, nullptr, &extCount, extProps.data());

	std::vector<const char*> devExts;
	if (m_surface) {
		devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	}
	// VK_KHR_portability_subset is a provisional extension; its name is only
	// declared in vulkan_beta.h, so use the string literal.
	if (RTXHasExt(extProps, "VK_KHR_portability_subset")) {
		// Required when a portability-subset device (MoltenVK) is used via an
		// instance created with ENUMERATE_PORTABILITY.
		devExts.push_back("VK_KHR_portability_subset");
	}
	if (m_rtSupported) {
		devExts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		devExts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
		devExts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	}

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = m_queueFamily;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;

	VkPhysicalDeviceRayQueryFeaturesKHR rqFeat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat = {
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
	};
	VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceFeatures2 feats = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	if (m_rtSupported) {
		feats.pNext = &v12;
		v12.pNext = &asFeat;
		asFeat.pNext = &rqFeat;
		v12.bufferDeviceAddress = VK_TRUE;
		asFeat.accelerationStructure = VK_TRUE;
		rqFeat.rayQuery = VK_TRUE;
	}

	VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext = m_rtSupported ? (const void*) &feats : nullptr;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = (uint32_t) devExts.size();
	dci.ppEnabledExtensionNames = devExts.data();

	VK_CHECK(vkCreateDevice(m_phys, &dci, nullptr, &m_device));
	vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

	pfnGetBufferAddress = (PFN_vkGetBufferDeviceAddress) vkGetDeviceProcAddr(m_device, "vkGetBufferDeviceAddress");
	if (!pfnGetBufferAddress) {
		pfnGetBufferAddress = (PFN_vkGetBufferDeviceAddress) vkGetDeviceProcAddr(m_device, "vkGetBufferDeviceAddressKHR");
	}
	if (m_rtSupported) {
		pfnCreateAS = (PFN_vkCreateAccelerationStructureKHR) vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR");
		pfnDestroyAS = (PFN_vkDestroyAccelerationStructureKHR) vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR");
		pfnGetASBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR) vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR");
		pfnCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR) vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR");
		pfnGetASAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR) vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR");
		if (!pfnCreateAS || !pfnDestroyAS || !pfnGetASBuildSizes || !pfnCmdBuildAS || !pfnGetASAddr) {
			m_rtSupported = false;
			m_rtEnabled = false;
		}
	}

	VkCommandPoolCreateInfo pool = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pool.queueFamilyIndex = m_queueFamily;
	if (vkCreateCommandPool(m_device, &pool, nullptr, &m_immediatePool) != VK_SUCCESS) {
		return false;
	}
	VkSemaphoreCreateInfo sem = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	if (vkCreateSemaphore(m_device, &sem, nullptr, &m_acquireSem) != VK_SUCCESS) {
		return false;
	}

	SDL_Log(
		"RTX renderer: %s, ray tracing %s",
		m_deviceName,
		m_rtEnabled ? "enabled" : (m_rtSupported ? "supported but disabled" : "unavailable")
	);
	return true;
}

bool Direct3DRMRTXRenderer::InitSwapchain()
{
	if (!m_surface) {
		return true; // headless mode
	}

	VkSurfaceCapabilitiesKHR caps;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_phys, m_surface, &caps));

	uint32_t fmtCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(m_phys, m_surface, &fmtCount, nullptr);
	if (!fmtCount) {
		return false;
	}
	std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(m_phys, m_surface, &fmtCount, fmts.data());
	m_swapFormat = fmts[0].format;
	for (auto& f : fmts) {
		if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
			m_swapFormat = f.format;
			break;
		}
	}

	m_swapExtent = caps.currentExtent;
	if (m_swapExtent.width == UINT32_MAX) {
		m_swapExtent = {(uint32_t) m_width, (uint32_t) m_height};
	}

	uint32_t imageCount = caps.minImageCount + 1;
	if (caps.maxImageCount && imageCount > caps.maxImageCount) {
		imageCount = caps.maxImageCount;
	}

	VkSwapchainCreateInfoKHR sci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
	sci.surface = m_surface;
	sci.minImageCount = imageCount;
	sci.imageFormat = m_swapFormat;
	sci.imageColorSpace = fmts[0].colorSpace;
	sci.imageExtent = m_swapExtent;
	sci.imageArrayLayers = 1;
	sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	sci.preTransform = caps.currentTransform;
	sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	sci.clipped = VK_TRUE;

	VK_CHECK(vkCreateSwapchainKHR(m_device, &sci, nullptr, &m_swapchain));

	uint32_t count = 0;
	vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, nullptr);
	m_swapImages.resize(count);
	vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, m_swapImages.data());
	m_swapViews.resize(count);
	for (uint32_t i = 0; i < count; ++i) {
		VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		vci.image = m_swapImages[i];
		vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
		vci.format = m_swapFormat;
		vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_swapViews[i]));
	}
	return true;
}

void Direct3DRMRTXRenderer::DestroySwapchain()
{
	for (auto v : m_swapViews) {
		vkDestroyImageView(m_device, v, nullptr);
	}
	m_swapViews.clear();
	m_swapImages.clear();
	if (m_swapchain) {
		vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
		m_swapchain = VK_NULL_HANDLE;
	}
}

bool Direct3DRMRTXRenderer::InitTargets()
{
	if (!CreateImage2D(
			m_width,
			m_height,
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			m_hdrImage,
			m_hdrMemory,
			m_hdrView
		)) {
		return false;
	}
	if (!CreateImage2D(
			m_width,
			m_height,
			m_swapFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			m_ldrImage,
			m_ldrMemory,
			m_ldrView
		)) {
		return false;
	}
	if (!CreateImage2D(
			m_width,
			m_height,
			VK_FORMAT_D32_SFLOAT,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			m_depthImage,
			m_depthMemory,
			m_depthView
		)) {
		return false;
	}

	{
		VkAttachmentDescription att[2] = {};
		att[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
		att[0].samples = VK_SAMPLE_COUNT_1_BIT;
		att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		att[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		att[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		att[1].format = VK_FORMAT_D32_SFLOAT;
		att[1].samples = VK_SAMPLE_COUNT_1_BIT;
		att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		att[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		att[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
		VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
		VkSubpassDescription sub = {};
		sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		sub.colorAttachmentCount = 1;
		sub.pColorAttachments = &colorRef;
		sub.pDepthStencilAttachment = &depthRef;
		VkRenderPassCreateInfo rpci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
		rpci.attachmentCount = 2;
		rpci.pAttachments = att;
		rpci.subpassCount = 1;
		rpci.pSubpasses = &sub;
		if (vkCreateRenderPass(m_device, &rpci, nullptr, &m_hdrPass) != VK_SUCCESS) {
			return false;
		}
	}
	{
		VkAttachmentDescription att = {};
		att.format = m_swapFormat;
		att.samples = VK_SAMPLE_COUNT_1_BIT;
		att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		att.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
		VkSubpassDescription sub = {};
		sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		sub.colorAttachmentCount = 1;
		sub.pColorAttachments = &colorRef;
		VkRenderPassCreateInfo rpci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
		rpci.attachmentCount = 1;
		rpci.pAttachments = &att;
		rpci.subpassCount = 1;
		rpci.pSubpasses = &sub;
		if (vkCreateRenderPass(m_device, &rpci, nullptr, &m_ldrPass) != VK_SUCCESS) {
			return false;
		}
	}

	VkImageView hdrAtt[2] = {m_hdrView, m_depthView};
	VkFramebufferCreateInfo fbci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
	fbci.renderPass = m_hdrPass;
	fbci.attachmentCount = 2;
	fbci.pAttachments = hdrAtt;
	fbci.width = m_width;
	fbci.height = m_height;
	fbci.layers = 1;
	if (vkCreateFramebuffer(m_device, &fbci, nullptr, &m_hdrFB) != VK_SUCCESS) {
		return false;
	}
	fbci.renderPass = m_ldrPass;
	fbci.attachmentCount = 1;
	fbci.pAttachments = &m_ldrView;
	if (vkCreateFramebuffer(m_device, &fbci, nullptr, &m_ldrFB) != VK_SUCCESS) {
		return false;
	}

	// Transition targets to their steady-state layouts.
	VkCommandBuffer cmd = BeginImmediate();
	VkImageMemoryBarrier barriers[3] = {};
	for (int i = 0; i < 3; ++i) {
		barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	}
	barriers[0].image = m_hdrImage;
	barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barriers[1].image = m_ldrImage;
	barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barriers[2].image = m_depthImage;
	barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	barriers[2].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	barriers[2].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		0,
		0,
		nullptr,
		0,
		nullptr,
		3,
		barriers
	);
	SubmitImmediate(cmd, {}, {});
	vkQueueWaitIdle(m_queue);

	// Dummy 1x1 white texture for untextured draws.
	uint32_t white = 0xFFFFFFFF;
	if (!CreateImage2D(
			1,
			1,
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			m_dummyImage,
			m_dummyMemory,
			m_dummyView
		)) {
		return false;
	}
	{
		VkCommandBuffer up = BeginImmediate();
		VkImageMemoryBarrier pre = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		pre.image = m_dummyImage;
		pre.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(up, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &pre);
		VkBuffer staging;
		VkDeviceMemory stagingMem;
		void* mapped;
		CreateBuffer(
			4,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			staging,
			stagingMem,
			true,
			&mapped
		);
		memcpy(mapped, &white, 4);
		vkUnmapMemory(m_device, stagingMem);
		VkBufferImageCopy copy = {};
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent = {1, 1, 1};
		vkCmdCopyBufferToImage(up, staging, m_dummyImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		VkImageMemoryBarrier post = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		post.image = m_dummyImage;
		post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(up, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &post);
		SubmitImmediate(up, {staging}, {stagingMem});
	}
	return true;
}

void Direct3DRMRTXRenderer::DestroyTargets()
{
	if (m_hdrFB) {
		vkDestroyFramebuffer(m_device, m_hdrFB, nullptr);
		m_hdrFB = VK_NULL_HANDLE;
	}
	if (m_ldrFB) {
		vkDestroyFramebuffer(m_device, m_ldrFB, nullptr);
		m_ldrFB = VK_NULL_HANDLE;
	}
	if (m_hdrPass) {
		vkDestroyRenderPass(m_device, m_hdrPass, nullptr);
		m_hdrPass = VK_NULL_HANDLE;
	}
	if (m_ldrPass) {
		vkDestroyRenderPass(m_device, m_ldrPass, nullptr);
		m_ldrPass = VK_NULL_HANDLE;
	}
	if (m_hdrView) {
		vkDestroyImageView(m_device, m_hdrView, nullptr);
		m_hdrView = VK_NULL_HANDLE;
	}
	if (m_ldrView) {
		vkDestroyImageView(m_device, m_ldrView, nullptr);
		m_ldrView = VK_NULL_HANDLE;
	}
	if (m_depthView) {
		vkDestroyImageView(m_device, m_depthView, nullptr);
		m_depthView = VK_NULL_HANDLE;
	}
	if (m_hdrImage) {
		vkDestroyImage(m_device, m_hdrImage, nullptr);
		vkFreeMemory(m_device, m_hdrMemory, nullptr);
		m_hdrImage = VK_NULL_HANDLE;
	}
	if (m_ldrImage) {
		vkDestroyImage(m_device, m_ldrImage, nullptr);
		vkFreeMemory(m_device, m_ldrMemory, nullptr);
		m_ldrImage = VK_NULL_HANDLE;
	}
	if (m_depthImage) {
		vkDestroyImage(m_device, m_depthImage, nullptr);
		vkFreeMemory(m_device, m_depthMemory, nullptr);
		m_depthImage = VK_NULL_HANDLE;
	}
}

// ---------------------------------------------------------------------------
// Pipelines and descriptors
// ---------------------------------------------------------------------------

bool Direct3DRMRTXRenderer::InitPipelines()
{
	{
		VkDescriptorSetLayoutBinding b = {};
		b.binding = 0;
		b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b.descriptorCount = 1;
		b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		ci.bindingCount = 1;
		ci.pBindings = &b;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_set0Plain));
	}
	{
		VkDescriptorSetLayoutBinding b[3] = {};
		b[0].binding = 0;
		b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b[0].descriptorCount = 1;
		b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		b[1].binding = 1;
		b[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		b[1].descriptorCount = 1;
		b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		b[2].binding = 2;
		b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		b[2].descriptorCount = 1;
		b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		ci.bindingCount = 3;
		ci.pBindings = b;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_set0RT));
	}
	{
		VkDescriptorSetLayoutBinding b[2] = {};
		b[0].binding = 0;
		b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b[0].descriptorCount = 1;
		b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		b[1].binding = 1;
		b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b[1].descriptorCount = 1;
		b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		ci.bindingCount = 2;
		ci.pBindings = b;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_set1));
	}
	{
		VkDescriptorSetLayoutBinding b = {};
		b.binding = 0;
		b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b.descriptorCount = 1;
		b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		ci.bindingCount = 1;
		ci.pBindings = &b;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_postSet0));
	}

	VkDescriptorSetLayout sceneSet0 = m_rtEnabled ? m_set0RT : m_set0Plain;
	VkDescriptorSetLayout sceneSets[2] = {sceneSet0, m_set1};
	VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	plci.setLayoutCount = 2;
	plci.pSetLayouts = sceneSets;
	VK_CHECK(vkCreatePipelineLayout(m_device, &plci, nullptr, &m_sceneLayout));

	VkDescriptorSetLayout postSets[1] = {m_postSet0};
	VkPushConstantRange pushRange = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
	plci.setLayoutCount = 1;
	plci.pSetLayouts = postSets;
	plci.pushConstantRangeCount = 1;
	plci.pPushConstantRanges = &pushRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &plci, nullptr, &m_postLayout));

	VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	sci.magFilter = VK_FILTER_LINEAR;
	sci.minFilter = VK_FILTER_LINEAR;
	sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	VK_CHECK(vkCreateSampler(m_device, &sci, nullptr, &m_sampler));
	sci.magFilter = VK_FILTER_NEAREST;
	sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	VK_CHECK(vkCreateSampler(m_device, &sci, nullptr, &m_uiSampler));

	VkDescriptorPoolSize poolSizes[4] = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16},
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
	};
	VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpci.maxSets = RTX_FRAMES_IN_FLIGHT + 1;
	dpci.poolSizeCount = 4;
	dpci.pPoolSizes = poolSizes;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpci, nullptr, &m_set0Pool));

	const uint32_t* fragSpv = m_rtEnabled ? scene_rt_frag_spv : scene_frag_spv;
	size_t fragSize = m_rtEnabled ? scene_rt_frag_spv_size : scene_frag_spv_size;
	m_opaquePipeline = CreateScenePipeline(fragSpv, fragSize, false, true);
	m_transparentPipeline = CreateScenePipeline(fragSpv, fragSize, true, false);
	m_uiPipeline = CreateUIPipeline();
	m_postPipeline = CreatePostPipeline();
	if (!m_opaquePipeline || !m_transparentPipeline || !m_uiPipeline || !m_postPipeline) {
		return false;
	}
	return true;
}

VkShaderModule Direct3DRMRTXRenderer::CreateShaderModule(const uint32_t* code, size_t size)
{
	VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	ci.codeSize = size;
	ci.pCode = code;
	VkShaderModule mod = VK_NULL_HANDLE;
	if (vkCreateShaderModule(m_device, &ci, nullptr, &mod) != VK_SUCCESS) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "vkCreateShaderModule failed");
	}
	return mod;
}

static void RTXFixedVertexInput(VkVertexInputBindingDescription& bind, VkVertexInputAttributeDescription* attr)
{
	bind = {0, sizeof(D3DRMVERTEX), VK_VERTEX_INPUT_RATE_VERTEX};
	attr[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(D3DRMVERTEX, position)};
	attr[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(D3DRMVERTEX, normal)};
	attr[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(D3DRMVERTEX, texCoord)};
}

VkPipeline Direct3DRMRTXRenderer::CreateScenePipeline(
	const uint32_t* fragSpv,
	size_t fragSize,
	bool blend,
	bool depthWrite
)
{
	VkShaderModule vs = CreateShaderModule(scene_vert_spv, scene_vert_spv_size);
	VkShaderModule fs = CreateShaderModule(fragSpv, fragSize);
	if (!vs || !fs) {
		return VK_NULL_HANDLE;
	}

	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vs;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fs;
	stages[1].pName = "main";

	VkVertexInputBindingDescription bind;
	VkVertexInputAttributeDescription attr[3];
	RTXFixedVertexInput(bind, attr);
	VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	vi.vertexBindingDescriptionCount = 1;
	vi.pVertexBindingDescriptions = &bind;
	vi.vertexAttributeDescriptionCount = 3;
	vi.pVertexAttributeDescriptions = attr;

	VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vp.viewportCount = 1;
	vp.scissorCount = 1;
	VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.polygonMode = VK_POLYGON_MODE_FILL;
	rs.cullMode = VK_CULL_MODE_BACK_BIT;
	rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rs.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
	ds.depthTestEnable = VK_TRUE;
	ds.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
	ds.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendAttachmentState cb = {};
	cb.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
						VK_COLOR_COMPONENT_A_BIT;
	cb.blendEnable = blend ? VK_TRUE : VK_FALSE;
	cb.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	cb.colorBlendOp = VK_BLEND_OP_ADD;
	cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	cb.alphaBlendOp = VK_BLEND_OP_ADD;
	VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cbs.attachmentCount = 1;
	cbs.pAttachments = &cb;

	VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dyn = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	dyn.dynamicStateCount = 2;
	dyn.pDynamicStates = dynStates;

	VkGraphicsPipelineCreateInfo pci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	pci.stageCount = 2;
	pci.pStages = stages;
	pci.pVertexInputState = &vi;
	pci.pInputAssemblyState = &ia;
	pci.pViewportState = &vp;
	pci.pRasterizationState = &rs;
	pci.pMultisampleState = &ms;
	pci.pDepthStencilState = &ds;
	pci.pColorBlendState = &cbs;
	pci.pDynamicState = &dyn;
	pci.layout = m_sceneLayout;
	pci.renderPass = m_hdrPass;
	pci.subpass = 0;

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
	vkDestroyShaderModule(m_device, vs, nullptr);
	vkDestroyShaderModule(m_device, fs, nullptr);
	if (res != VK_SUCCESS) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "vkCreateGraphicsPipelines (scene) failed: %d", (int) res);
	}
	return pipeline;
}

VkPipeline Direct3DRMRTXRenderer::CreateUIPipeline()
{
	VkShaderModule vs = CreateShaderModule(ui_vert_spv, ui_vert_spv_size);
	VkShaderModule fs = CreateShaderModule(ui_frag_spv, ui_frag_spv_size);
	if (!vs || !fs) {
		return VK_NULL_HANDLE;
	}
	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vs;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fs;
	stages[1].pName = "main";

	VkVertexInputBindingDescription bind;
	VkVertexInputAttributeDescription attr[3];
	RTXFixedVertexInput(bind, attr);
	VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	vi.vertexBindingDescriptionCount = 1;
	vi.pVertexBindingDescriptions = &bind;
	vi.vertexAttributeDescriptionCount = 3;
	vi.pVertexAttributeDescriptions = attr;

	VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vp.viewportCount = 1;
	vp.scissorCount = 1;
	VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.polygonMode = VK_POLYGON_MODE_FILL;
	rs.cullMode = VK_CULL_MODE_NONE;
	rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rs.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

	VkPipelineColorBlendAttachmentState cb = {};
	cb.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
						VK_COLOR_COMPONENT_A_BIT;
	cb.blendEnable = VK_TRUE;
	cb.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	cb.colorBlendOp = VK_BLEND_OP_ADD;
	cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	cb.alphaBlendOp = VK_BLEND_OP_ADD;
	VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cbs.attachmentCount = 1;
	cbs.pAttachments = &cb;

	VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dyn = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	dyn.dynamicStateCount = 2;
	dyn.pDynamicStates = dynStates;

	VkGraphicsPipelineCreateInfo pci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	pci.stageCount = 2;
	pci.pStages = stages;
	pci.pVertexInputState = &vi;
	pci.pInputAssemblyState = &ia;
	pci.pViewportState = &vp;
	pci.pRasterizationState = &rs;
	pci.pMultisampleState = &ms;
	pci.pDepthStencilState = &ds;
	pci.pColorBlendState = &cbs;
	pci.pDynamicState = &dyn;
	pci.layout = m_sceneLayout;
	pci.renderPass = m_ldrPass;
	pci.subpass = 0;

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
	vkDestroyShaderModule(m_device, vs, nullptr);
	vkDestroyShaderModule(m_device, fs, nullptr);
	if (res != VK_SUCCESS) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "vkCreateGraphicsPipelines (ui) failed: %d", (int) res);
	}
	return pipeline;
}

VkPipeline Direct3DRMRTXRenderer::CreatePostPipeline()
{
	VkShaderModule vs = CreateShaderModule(post_vert_spv, post_vert_spv_size);
	VkShaderModule fs = CreateShaderModule(post_frag_spv, post_frag_spv_size);
	if (!vs || !fs) {
		return VK_NULL_HANDLE;
	}
	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vs;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fs;
	stages[1].pName = "main";

	VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vp.viewportCount = 1;
	vp.scissorCount = 1;
	VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.polygonMode = VK_POLYGON_MODE_FILL;
	rs.cullMode = VK_CULL_MODE_NONE;
	rs.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
	VkPipelineColorBlendAttachmentState cb = {};
	cb.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
						VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cbs.attachmentCount = 1;
	cbs.pAttachments = &cb;
	VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dyn = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	dyn.dynamicStateCount = 2;
	dyn.pDynamicStates = dynStates;

	VkGraphicsPipelineCreateInfo pci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	pci.stageCount = 2;
	pci.pStages = stages;
	pci.pVertexInputState = &vi;
	pci.pInputAssemblyState = &ia;
	pci.pViewportState = &vp;
	pci.pRasterizationState = &rs;
	pci.pMultisampleState = &ms;
	pci.pDepthStencilState = &ds;
	pci.pColorBlendState = &cbs;
	pci.pDynamicState = &dyn;
	pci.layout = m_postLayout;
	pci.renderPass = m_ldrPass;
	pci.subpass = 0;

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
	vkDestroyShaderModule(m_device, vs, nullptr);
	vkDestroyShaderModule(m_device, fs, nullptr);
	if (res != VK_SUCCESS) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "vkCreateGraphicsPipelines (post) failed: %d", (int) res);
	}
	return pipeline;
}

// ---------------------------------------------------------------------------
// Frame slots
// ---------------------------------------------------------------------------

bool Direct3DRMRTXRenderer::InitFrameSlots()
{
	for (auto& slot : m_slots) {
		VkCommandPoolCreateInfo pool = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool.queueFamilyIndex = m_queueFamily;
		VK_CHECK(vkCreateCommandPool(m_device, &pool, nullptr, &slot.cmdPool));
		VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cai.commandPool = slot.cmdPool;
		cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cai.commandBufferCount = 1;
		VK_CHECK(vkAllocateCommandBuffers(m_device, &cai, &slot.cmd));
		VkFenceCreateInfo fence = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		VK_CHECK(vkCreateFence(m_device, &fence, nullptr, &slot.fence));
		VkSemaphoreCreateInfo sem = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VK_CHECK(vkCreateSemaphore(m_device, &sem, nullptr, &slot.renderDone));

		if (!CreateBuffer(
				sizeof(RTXFrameUBO),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				slot.frameUBO,
				slot.frameUBOMemory,
				true,
				&slot.frameUBOMap
			)) {
			return false;
		}
		if (!CreateBuffer(
				RTX_DRAW_UBO_STRIDE * RTX_MAX_DRAWS,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				slot.drawUBO,
				slot.drawUBOMemory,
				true,
				&slot.drawUBOMap
			)) {
			return false;
		}
		if (m_rtEnabled) {
			if (!CreateBuffer(
					64 * sizeof(RTXInstanceInfo),
					VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					slot.instSSBO,
					slot.instSSBOMemory,
					true,
					&slot.instSSBOMap
				)) {
				return false;
			}
			slot.instSSBOSize = 64 * sizeof(RTXInstanceInfo);
		}

		VkDescriptorPoolSize sizes[3] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, RTX_MAX_DRAWS},
			{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, RTX_MAX_DRAWS},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
		};
		VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		dpci.maxSets = RTX_MAX_DRAWS;
		dpci.poolSizeCount = 3;
		dpci.pPoolSizes = sizes;
		VK_CHECK(vkCreateDescriptorPool(m_device, &dpci, nullptr, &slot.drawDescPool));

		VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		dsai.descriptorPool = m_set0Pool;
		dsai.descriptorSetCount = 1;
		VkDescriptorSetLayout layout = m_rtEnabled ? m_set0RT : m_set0Plain;
		dsai.pSetLayouts = &layout;
		VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &slot.set0));
	}

	if (m_rtEnabled && !CreateDummyTLAS()) {
		return false;
	}

	// Post descriptor set (HDR input).
	{
		VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		dsai.descriptorPool = m_set0Pool;
		dsai.descriptorSetCount = 1;
		dsai.pSetLayouts = &m_postSet0;
		VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_postSet));
		VkDescriptorImageInfo ii = {m_sampler, m_hdrView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		w.dstSet = m_postSet;
		w.dstBinding = 0;
		w.descriptorCount = 1;
		w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		w.pImageInfo = &ii;
		vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
	}

	// UI quad mesh
	m_uiMesh.vertices = {
		{{0.0f, 0.0f, 0.0f}, {0, 0, -1}, {0.0f, 0.0f}},
		{{1.0f, 0.0f, 0.0f}, {0, 0, -1}, {1.0f, 0.0f}},
		{{1.0f, 1.0f, 0.0f}, {0, 0, -1}, {1.0f, 1.0f}},
		{{0.0f, 1.0f, 0.0f}, {0, 0, -1}, {0.0f, 1.0f}}
	};
	m_uiMesh.indices = {0, 1, 2, 0, 2, 3};
	m_uiMeshCache = UploadMeshData(m_uiMesh, false);
	if (!m_uiMeshCache.vertexBuffer) {
		return false;
	}
	for (auto& slot : m_slots) {
		UpdateSet0(slot);
	}
	return true;
}

void Direct3DRMRTXRenderer::UpdateSet0(RTXFrameSlot& slot)
{
	VkDescriptorBufferInfo ubo = {slot.frameUBO, 0, sizeof(RTXFrameUBO)};
	VkWriteDescriptorSet writes[3] = {};
	uint32_t n = 0;
	writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[n].dstSet = slot.set0;
	writes[n].dstBinding = 0;
	writes[n].descriptorCount = 1;
	writes[n].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[n].pBufferInfo = &ubo;
	n++;
	VkWriteDescriptorSetAccelerationStructureKHR asWrite = {
		VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR
	};
	VkDescriptorBufferInfo ssbo = {};
	if (m_rtEnabled && m_tlas && slot.instSSBO) {
		asWrite.accelerationStructureCount = 1;
		asWrite.pAccelerationStructures = &m_tlas;
		writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[n].pNext = &asWrite;
		writes[n].dstSet = slot.set0;
		writes[n].dstBinding = 1;
		writes[n].descriptorCount = 1;
		writes[n].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		n++;
		ssbo = {slot.instSSBO, 0, VK_WHOLE_SIZE};
		writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[n].dstSet = slot.set0;
		writes[n].dstBinding = 2;
		writes[n].descriptorCount = 1;
		writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[n].pBufferInfo = &ssbo;
		n++;
	}
	vkUpdateDescriptorSets(m_device, n, writes, 0, nullptr);
}

void Direct3DRMRTXRenderer::WaitForSlot(RTXFrameSlot& slot)
{
	if (!slot.submitted) {
		return;
	}
	vkWaitForFences(m_device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
	slot.submitted = false;
	for (auto b : slot.trashBuffers) {
		vkDestroyBuffer(m_device, b, nullptr);
	}
	for (auto m : slot.trashMemories) {
		vkFreeMemory(m_device, m, nullptr);
	}
	slot.trashBuffers.clear();
	slot.trashMemories.clear();
}

// ---------------------------------------------------------------------------
// Resource helpers
// ---------------------------------------------------------------------------

uint32_t Direct3DRMRTXRenderer::FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props)
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(m_phys, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
		if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
			return i;
		}
	}
	SDL_LogError(LOG_CATEGORY_MINIWIN, "No matching memory type (bits %x props %x)", typeBits, (unsigned) props);
	return UINT32_MAX;
}

bool Direct3DRMRTXRenderer::CreateBuffer(
	VkDeviceSize size,
	VkBufferUsageFlags usage,
	VkMemoryPropertyFlags props,
	VkBuffer& buffer,
	VkDeviceMemory& memory,
	bool hostMapped,
	void** mapped
)
{
	VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size = size;
	bci.usage = usage;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (vkCreateBuffer(m_device, &bci, nullptr, &buffer) != VK_SUCCESS) {
		return false;
	}
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(m_device, buffer, &req);
	uint32_t type = FindMemoryType(req.memoryTypeBits, props);
	if (type == UINT32_MAX) {
		return false;
	}
	VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = type;
	VkMemoryAllocateFlagsInfo flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
		flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		mai.pNext = &flags;
	}
	if (vkAllocateMemory(m_device, &mai, nullptr, &memory) != VK_SUCCESS) {
		return false;
	}
	vkBindBufferMemory(m_device, buffer, memory, 0);
	if (hostMapped && mapped) {
		vkMapMemory(m_device, memory, 0, size, 0, mapped);
	}
	return true;
}

bool Direct3DRMRTXRenderer::CreateImage2D(
	uint32_t w,
	uint32_t h,
	VkFormat format,
	VkImageUsageFlags usage,
	VkImage& image,
	VkDeviceMemory& memory,
	VkImageView& view
)
{
	VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ici.imageType = VK_IMAGE_TYPE_2D;
	ici.format = format;
	ici.extent = {w, h, 1};
	ici.mipLevels = 1;
	ici.arrayLayers = 1;
	ici.samples = VK_SAMPLE_COUNT_1_BIT;
	ici.tiling = VK_IMAGE_TILING_OPTIMAL;
	ici.usage = usage;
	ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (vkCreateImage(m_device, &ici, nullptr, &image) != VK_SUCCESS) {
		return false;
	}
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(m_device, image, &req);
	uint32_t type = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (type == UINT32_MAX) {
		return false;
	}
	VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = type;
	if (vkAllocateMemory(m_device, &mai, nullptr, &memory) != VK_SUCCESS) {
		return false;
	}
	vkBindImageMemory(m_device, image, memory, 0);
	VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vci.image = image;
	vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vci.format = format;
	vci.subresourceRange.aspectMask =
		(format == VK_FORMAT_D32_SFLOAT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	vci.subresourceRange.levelCount = 1;
	vci.subresourceRange.layerCount = 1;
	if (vkCreateImageView(m_device, &vci, nullptr, &view) != VK_SUCCESS) {
		return false;
	}
	return true;
}

VkDeviceAddress Direct3DRMRTXRenderer::BufferAddress(VkBuffer buffer)
{
	VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
	info.buffer = buffer;
	return pfnGetBufferAddress(m_device, &info);
}

VkCommandBuffer Direct3DRMRTXRenderer::BeginImmediate()
{
	VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cai.commandPool = m_immediatePool;
	cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cai.commandBufferCount = 1;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(m_device, &cai, &cmd);
	VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &bi);
	return cmd;
}

void Direct3DRMRTXRenderer::SubmitImmediate(
	VkCommandBuffer cmd,
	std::vector<VkBuffer> bufs,
	std::vector<VkDeviceMemory> mems
)
{
	vkEndCommandBuffer(cmd);
	RTXPendingOp op;
	op.cmd = cmd;
	op.buffers = std::move(bufs);
	op.memories = std::move(mems);
	VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	vkCreateFence(m_device, &fci, nullptr, &op.fence);
	VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	vkQueueSubmit(m_queue, 1, &si, op.fence);
	m_pendingOps.push_back(std::move(op));
	ReapPending();
}

void Direct3DRMRTXRenderer::ReapPending()
{
	for (auto it = m_pendingOps.begin(); it != m_pendingOps.end();) {
		if (!it->fence || vkGetFenceStatus(m_device, it->fence) != VK_SUCCESS) {
			++it;
			continue;
		}
		if (it->fence) {
			vkDestroyFence(m_device, it->fence, nullptr);
		}
		if (it->cmd) {
			vkFreeCommandBuffers(m_device, m_immediatePool, 1, &it->cmd);
		}
		for (auto b : it->buffers) {
			vkDestroyBuffer(m_device, b, nullptr);
		}
		for (auto m : it->memories) {
			vkFreeMemory(m_device, m, nullptr);
		}
		for (auto as : it->structures) {
			if (pfnDestroyAS) {
				pfnDestroyAS(m_device, as, nullptr);
			}
		}
		it = m_pendingOps.erase(it);
	}
}

void Direct3DRMRTXRenderer::RetireBuffer(VkBuffer b, VkDeviceMemory m)
{
	if (!m_device || (!b && !m)) {
		return;
	}
	// The buffer may still be referenced by in-flight work. An empty submit
	// with a fence signals only after everything queued so far completes,
	// making it a safe destruction point.
	RTXPendingOp op;
	op.buffers.push_back(b);
	op.memories.push_back(m);
	VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	vkCreateFence(m_device, &fci, nullptr, &op.fence);
	VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	vkQueueSubmit(m_queue, 1, &si, op.fence);
	m_pendingOps.push_back(std::move(op));
}

void Direct3DRMRTXRenderer::RetireAS(VkAccelerationStructureKHR as, VkBuffer b, VkDeviceMemory m)
{
	if (!m_device || (!as && !b && !m)) {
		return;
	}
	RTXPendingOp op;
	op.structures.push_back(as);
	op.buffers.push_back(b);
	op.memories.push_back(m);
	VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	vkCreateFence(m_device, &fci, nullptr, &op.fence);
	VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	vkQueueSubmit(m_queue, 1, &si, op.fence);
	m_pendingOps.push_back(std::move(op));
}

// ---------------------------------------------------------------------------
// Acceleration structures
// ---------------------------------------------------------------------------

bool Direct3DRMRTXRenderer::CreateBLAS(RTXMeshCache& mesh)
{
	if (!m_rtEnabled || !mesh.indexCount || !mesh.vertexBuffer) {
		return true;
	}

	VkAccelerationStructureGeometryTrianglesDataKHR tris = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
	};
	tris.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	tris.vertexData.deviceAddress = BufferAddress(mesh.vertexBuffer);
	tris.vertexStride = sizeof(D3DRMVERTEX);
	tris.maxVertex = mesh.vertexCount;
	tris.indexType = VK_INDEX_TYPE_UINT16;
	tris.indexData.deviceAddress = BufferAddress(mesh.indexBuffer);

	VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
	geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geo.geometry.triangles = tris;
	geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

	VkAccelerationStructureBuildGeometryInfoKHR build = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
	};
	build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.geometryCount = 1;
	build.pGeometries = &geo;

	uint32_t primCount = mesh.indexCount / 3;
	VkAccelerationStructureBuildSizesInfoKHR sizes = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
	};
	pfnGetASBuildSizes(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &primCount, &sizes);

	if (!CreateBuffer(
			sizes.accelerationStructureSize,
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			mesh.blasBuffer,
			mesh.blasMemory
		)) {
		return false;
	}
	VkAccelerationStructureCreateInfoKHR asci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
	asci.buffer = mesh.blasBuffer;
	asci.size = sizes.accelerationStructureSize;
	asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	if (pfnCreateAS(m_device, &asci, nullptr, &mesh.blas) != VK_SUCCESS) {
		return false;
	}

	VkBuffer scratch;
	VkDeviceMemory scratchMem;
	if (!CreateBuffer(
			sizes.buildScratchSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			scratch,
			scratchMem
		)) {
		return false;
	}
	build.dstAccelerationStructure = mesh.blas;
	build.scratchData.deviceAddress = BufferAddress(scratch);

	VkAccelerationStructureBuildRangeInfoKHR range = {};
	range.primitiveCount = primCount;
	const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};

	VkCommandBuffer cmd = BeginImmediate();
	pfnCmdBuildAS(cmd, 1, &build, ranges);
	SubmitImmediate(cmd, {scratch}, {scratchMem});
	return true;
}

bool Direct3DRMRTXRenderer::BuildTLAS(RTXFrameSlot& slot, VkCommandBuffer cmd)
{
	size_t instCount = m_draws.size() + m_transparentDraws.size();
	if (!instCount) {
		return true;
	}
	std::vector<VkAccelerationStructureInstanceKHR> instances(instCount);
	size_t idx = 0;
	auto fill = [&](const std::vector<RTXDraw>& draws, bool transparent) {
		for (size_t i = 0; i < draws.size(); ++i, ++idx) {
			auto& inst = instances[idx];
			memset(&inst, 0, sizeof(inst));
			const D3DRMMATRIX4D& w = draws[i].worldMatrix;
			// VkTransformMatrixKHR is a row-major 3x4 affine in column-vector
			// convention: the transpose of the engine's row-vector 4x4.
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 4; ++c) {
					inst.transform.matrix[r][c] = w[c][r];
				}
			}
			const RTXMeshCache& mesh = m_meshs[draws[i].meshId];
			VkAccelerationStructureDeviceAddressInfoKHR ai = {
				VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR
			};
			ai.accelerationStructure = mesh.blas;
			inst.accelerationStructureReference = pfnGetASAddr(m_device, &ai);
			inst.instanceCustomIndex = (uint32_t) idx;
			inst.mask = transparent ? 0x2 : 0x1;
			inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		}
	};
	fill(m_draws, false);
	fill(m_transparentDraws, true);

	VkDeviceSize instSize = sizeof(VkAccelerationStructureInstanceKHR) * instCount;
	if (slot.tlasInstanceSize < instSize) {
		if (slot.tlasInstanceBuffer) {
			RetireBuffer(slot.tlasInstanceBuffer, slot.tlasInstanceMemory);
			slot.tlasInstanceBuffer = VK_NULL_HANDLE;
		}
		if (!CreateBuffer(
				instSize,
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
					VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				slot.tlasInstanceBuffer,
				slot.tlasInstanceMemory
			)) {
			return false;
		}
		slot.tlasInstanceSize = instSize;
	}

	VkBuffer staging;
	VkDeviceMemory stagingMem;
	void* mapped = nullptr;
	if (!CreateBuffer(
			instSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			staging,
			stagingMem,
			true,
			&mapped
		)) {
		return false;
	}
	memcpy(mapped, instances.data(), instSize);
	vkUnmapMemory(m_device, stagingMem);
	slot.trashBuffers.push_back(staging);
	slot.trashMemories.push_back(stagingMem);

	VkBufferCopy copy = {0, 0, instSize};
	vkCmdCopyBuffer(cmd, staging, slot.tlasInstanceBuffer, 1, &copy);
	VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		0,
		1,
		&mb,
		0,
		nullptr,
		0,
		nullptr
	);

	VkAccelerationStructureGeometryInstancesDataKHR instData = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
	};
	instData.arrayOfPointers = VK_FALSE;
	instData.data.deviceAddress = BufferAddress(slot.tlasInstanceBuffer);

	VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
	geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geo.geometry.instances = instData;

	VkAccelerationStructureBuildGeometryInfoKHR build = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
	};
	build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.geometryCount = 1;
	build.pGeometries = &geo;

	uint32_t primCount = (uint32_t) instCount;
	VkAccelerationStructureBuildSizesInfoKHR sizes = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
	};
	pfnGetASBuildSizes(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &primCount, &sizes);

	if (m_tlasSize < sizes.accelerationStructureSize) {
		RetireAS(m_tlas, m_tlasBuffer, m_tlasMemory);
		m_tlas = VK_NULL_HANDLE;
		m_tlasSize = 0;
		VkBuffer buf;
		VkDeviceMemory mem;
		if (!CreateBuffer(
				sizes.accelerationStructureSize,
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
					VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				buf,
				mem
			)) {
			return false;
		}
		VkAccelerationStructureCreateInfoKHR asci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
		asci.buffer = buf;
		asci.size = sizes.accelerationStructureSize;
		asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		if (pfnCreateAS(m_device, &asci, nullptr, &m_tlas) != VK_SUCCESS) {
			vkDestroyBuffer(m_device, buf, nullptr);
			vkFreeMemory(m_device, mem, nullptr);
			return false;
		}
		m_tlasBuffer = buf;
		m_tlasMemory = mem;
		m_tlasSize = sizes.accelerationStructureSize;
	}

	VkBuffer scratch;
	VkDeviceMemory scratchMem;
	if (!CreateBuffer(
			sizes.buildScratchSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			scratch,
			scratchMem
		)) {
		return false;
	}
	slot.trashBuffers.push_back(scratch);
	slot.trashMemories.push_back(scratchMem);

	build.dstAccelerationStructure = m_tlas;
	build.scratchData.deviceAddress = BufferAddress(scratch);

	VkAccelerationStructureBuildRangeInfoKHR range = {};
	range.primitiveCount = primCount;
	const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};
	pfnCmdBuildAS(cmd, 1, &build, ranges);

	mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
	mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		1,
		&mb,
		0,
		nullptr,
		0,
		nullptr
	);
	return true;
}

bool Direct3DRMRTXRenderer::CreateDummyTLAS()
{
	// A 1-triangle BLAS + 1 invisible-instance TLAS so RT descriptors always
	// have a valid bound object, even before the first real frame.
	D3DRMVERTEX tri[3] = {
		{{0, 0, 0}, {0, 0, -1}, {0, 0}},
		{{1, 0, 0}, {0, 0, -1}, {1, 0}},
		{{0, 1, 0}, {0, 0, -1}, {0, 1}},
	};
	uint16_t idx[3] = {0, 1, 2};

	VkBuffer vb;
	VkDeviceMemory vbMem;
	CreateBuffer(
		sizeof(tri),
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		vb,
		vbMem
	);
	VkBuffer ib;
	VkDeviceMemory ibMem;
	CreateBuffer(
		sizeof(idx),
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		ib,
		ibMem
	);
	VkBuffer staging;
	VkDeviceMemory stagingMem;
	void* mapped;
	CreateBuffer(
		sizeof(tri) + sizeof(idx),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		staging,
		stagingMem,
		true,
		&mapped
	);
	memcpy(mapped, tri, sizeof(tri));
	memcpy((char*) mapped + sizeof(tri), idx, sizeof(idx));
	vkUnmapMemory(m_device, stagingMem);
	VkCommandBuffer cmd = BeginImmediate();
	VkBufferCopy c1 = {0, 0, sizeof(tri)};
	VkBufferCopy c2 = {sizeof(tri), 0, sizeof(idx)};
	vkCmdCopyBuffer(cmd, staging, vb, 1, &c1);
	vkCmdCopyBuffer(cmd, staging, ib, 1, &c2);
	SubmitImmediate(cmd, {staging}, {stagingMem});
	vkQueueWaitIdle(m_queue);
	ReapPending();

	RTXMeshCache dummyMesh = {};
	dummyMesh.vertexBuffer = vb;
	dummyMesh.indexBuffer = ib;
	dummyMesh.vertexCount = 3;
	dummyMesh.indexCount = 3;
	if (!CreateBLAS(dummyMesh)) {
		return false;
	}
	vkQueueWaitIdle(m_queue);
	ReapPending();
	m_dummyBLAS = dummyMesh.blas;
	m_dummyBLASBuffer = dummyMesh.blasBuffer;
	m_dummyBLASMemory = dummyMesh.blasMemory;
	// vb/ib stay alive permanently with the dummy BLAS.
	m_keepAliveBuffers.push_back(vb);
	m_keepAliveBuffers.push_back(ib);
	m_keepAliveMemories.push_back(vbMem);
	m_keepAliveMemories.push_back(ibMem);

	VkBuffer instBuf;
	VkDeviceMemory instMem;
	VkAccelerationStructureInstanceKHR inst = {};
	inst.transform.matrix[0][0] = 1.f;
	inst.transform.matrix[1][1] = 1.f;
	inst.transform.matrix[2][2] = 1.f;
	inst.mask = 0; // never intersected
	VkAccelerationStructureDeviceAddressInfoKHR ai = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
	ai.accelerationStructure = m_dummyBLAS;
	inst.accelerationStructureReference = pfnGetASAddr(m_device, &ai);

	CreateBuffer(
		sizeof(inst),
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		instBuf,
		instMem
	);
	CreateBuffer(
		sizeof(inst),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		staging,
		stagingMem,
		true,
		&mapped
	);
	memcpy(mapped, &inst, sizeof(inst));
	vkUnmapMemory(m_device, stagingMem);
	cmd = BeginImmediate();
	VkBufferCopy cc = {0, 0, sizeof(inst)};
	vkCmdCopyBuffer(cmd, staging, instBuf, 1, &cc);
	SubmitImmediate(cmd, {staging}, {stagingMem});
	vkQueueWaitIdle(m_queue);
	ReapPending();
	m_keepAliveBuffers.push_back(instBuf);
	m_keepAliveMemories.push_back(instMem);

	VkAccelerationStructureGeometryInstancesDataKHR instData = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
	};
	instData.data.deviceAddress = BufferAddress(instBuf);
	VkAccelerationStructureGeometryKHR geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
	geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geo.geometry.instances = instData;
	VkAccelerationStructureBuildGeometryInfoKHR build = {
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
	};
	build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build.geometryCount = 1;
	build.pGeometries = &geo;
	uint32_t primCount = 1;
	VkAccelerationStructureBuildSizesInfoKHR sizes = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
	pfnGetASBuildSizes(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &primCount, &sizes);

	CreateBuffer(
		sizes.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		m_tlasBuffer,
		m_tlasMemory
	);
	VkAccelerationStructureCreateInfoKHR asci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
	asci.buffer = m_tlasBuffer;
	asci.size = sizes.accelerationStructureSize;
	asci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	if (pfnCreateAS(m_device, &asci, nullptr, &m_tlas) != VK_SUCCESS) {
		return false;
	}
	m_tlasSize = sizes.accelerationStructureSize;

	VkBuffer scratch;
	VkDeviceMemory scratchMem;
	CreateBuffer(
		sizes.buildScratchSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		scratch,
		scratchMem
	);
	build.dstAccelerationStructure = m_tlas;
	build.scratchData.deviceAddress = BufferAddress(scratch);
	VkAccelerationStructureBuildRangeInfoKHR range = {};
	range.primitiveCount = 1;
	const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};
	cmd = BeginImmediate();
	pfnCmdBuildAS(cmd, 1, &build, ranges);
	SubmitImmediate(cmd, {scratch}, {scratchMem});
	vkQueueWaitIdle(m_queue);
	ReapPending();
	return true;
}

// ---------------------------------------------------------------------------
// Textures & meshes
// ---------------------------------------------------------------------------

Uint32 Direct3DRMRTXRenderer::GetTextureId(
	IDirect3DRMTexture* iTexture,
	bool isUI,
	float scaleX,
	float scaleY
)
{
	auto texture = static_cast<Direct3DRMTextureImpl*>(iTexture);
	auto surface = static_cast<DirectDrawSurfaceImpl*>(texture->m_surface);
	SDL_Surface* surf = surface->m_surface;

	for (Uint32 i = 0; i < m_textures.size(); ++i) {
		auto& tex = m_textures[i];
		if (tex.texture == texture) {
			if (tex.version != texture->m_version) {
				if (!UploadTexture(tex, surf)) {
					return NO_TEXTURE_ID;
				}
				tex.version = texture->m_version;
			}
			return i;
		}
	}

	RTXTextureCache entry;
	entry.texture = texture;
	entry.version = texture->m_version;
	if (!UploadTexture(entry, surf)) {
		return NO_TEXTURE_ID;
	}

	auto destroyCb = [](Direct3DRMRenderer* renderer, Uint32 id) {
		auto* self = static_cast<Direct3DRMRTXRenderer*>(renderer);
		auto& c = self->m_textures[id];
		if (c.image) {
			vkDestroyImageView(self->m_device, c.view, nullptr);
			vkDestroyImage(self->m_device, c.image, nullptr);
			vkFreeMemory(self->m_device, c.memory, nullptr);
			c = {};
		}
	};

	for (Uint32 i = 0; i < m_textures.size(); ++i) {
		if (!m_textures[i].texture) {
			m_textures[i] = entry;
			RegisterDestroyCallback(texture, i, destroyCb);
			return i;
		}
	}
	m_textures.push_back(entry);
	RegisterDestroyCallback(texture, (Uint32) (m_textures.size() - 1), destroyCb);
	return (Uint32) (m_textures.size() - 1);
}

bool Direct3DRMRTXRenderer::UploadTexture(RTXTextureCache& entry, SDL_Surface* surface)
{
	SDL_Surface* conv = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGBA32);
	if (!conv) {
		return false;
	}
	const uint32_t w = conv->w;
	const uint32_t h = conv->h;
	VkDeviceSize size = (VkDeviceSize) conv->pitch * conv->h;
	if (entry.image) {
		vkDeviceWaitIdle(m_device);
		vkDestroyImageView(m_device, entry.view, nullptr);
		vkDestroyImage(m_device, entry.image, nullptr);
		vkFreeMemory(m_device, entry.memory, nullptr);
		entry.image = VK_NULL_HANDLE;
	}
	if (!CreateImage2D(
			w,
			h,
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			entry.image,
			entry.memory,
			entry.view
		)) {
		SDL_DestroySurface(conv);
		return false;
	}

	VkBuffer staging;
	VkDeviceMemory stagingMem;
	void* mapped = nullptr;
	bool ok = CreateBuffer(
		size,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		staging,
		stagingMem,
		true,
		&mapped
	);
	if (ok) {
		memcpy(mapped, conv->pixels, size);
		vkUnmapMemory(m_device, stagingMem);

		VkCommandBuffer cmd = BeginImmediate();
		VkImageMemoryBarrier pre = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		pre.image = entry.image;
		pre.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&pre
		);
		VkBufferImageCopy copy = {};
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent = {w, h, 1};
		copy.bufferRowLength = 0;
		vkCmdCopyBufferToImage(cmd, staging, entry.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		VkImageMemoryBarrier post = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		post.image = entry.image;
		post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&post
		);
		SubmitImmediate(cmd, {staging}, {stagingMem});
	}
	SDL_DestroySurface(conv);
	return ok;
}

RTXMeshCache Direct3DRMRTXRenderer::UploadMeshData(const MeshGroup& meshGroup, bool buildBLAS)
{
	std::vector<D3DRMVERTEX> finalVertices;
	std::vector<Uint16> finalIndices;

	if (meshGroup.quality == D3DRMRENDER_FLAT || meshGroup.quality == D3DRMRENDER_UNLITFLAT) {
		std::vector<uint16_t> newIndices;
		FlattenSurfaces(
			meshGroup.vertices.data(),
			meshGroup.vertices.size(),
			meshGroup.indices.data(),
			meshGroup.indices.size(),
			true,
			finalVertices,
			newIndices
		);
		finalIndices.assign(newIndices.begin(), newIndices.end());
	}
	else {
		finalVertices = meshGroup.vertices;
		finalIndices.assign(meshGroup.indices.begin(), meshGroup.indices.end());
	}

	RTXMeshCache cache = {};
	cache.meshGroup = &meshGroup;
	cache.version = meshGroup.version;
	cache.indexCount = (uint32_t) finalIndices.size();
	cache.vertexCount = (uint32_t) finalVertices.size();
	if (!cache.indexCount || !cache.vertexCount) {
		return cache;
	}

	VkBufferUsageFlags vbUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags ibUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if (m_rtEnabled && buildBLAS) {
		vbUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
				   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
		ibUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
				   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	}
	if (!CreateBuffer(
			sizeof(D3DRMVERTEX) * finalVertices.size(),
			vbUsage,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			cache.vertexBuffer,
			cache.vertexMemory
		)) {
		return {};
	}
	if (!CreateBuffer(
			sizeof(Uint16) * finalIndices.size(),
			ibUsage,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			cache.indexBuffer,
			cache.indexMemory
		)) {
		vkDestroyBuffer(m_device, cache.vertexBuffer, nullptr);
		vkFreeMemory(m_device, cache.vertexMemory, nullptr);
		return {};
	}

	VkDeviceSize vsize = sizeof(D3DRMVERTEX) * finalVertices.size();
	VkDeviceSize isize = sizeof(Uint16) * finalIndices.size();
	VkBuffer staging;
	VkDeviceMemory stagingMem;
	void* mapped = nullptr;
	if (!CreateBuffer(
			vsize + isize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			staging,
			stagingMem,
			true,
			&mapped
		)) {
		return {};
	}
	memcpy(mapped, finalVertices.data(), vsize);
	memcpy((char*) mapped + vsize, finalIndices.data(), isize);
	vkUnmapMemory(m_device, stagingMem);

	VkCommandBuffer cmd = BeginImmediate();
	VkBufferCopy copies[2] = {{0, 0, vsize}, {vsize, 0, isize}};
	vkCmdCopyBuffer(cmd, staging, cache.vertexBuffer, 1, &copies[0]);
	vkCmdCopyBuffer(cmd, staging, cache.indexBuffer, 1, &copies[1]);
	VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
	if (m_rtEnabled) {
		mb.dstAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
	}
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		0,
		1,
		&mb,
		0,
		nullptr,
		0,
		nullptr
	);
	SubmitImmediate(cmd, {staging}, {stagingMem});

	if (buildBLAS) {
		CreateBLAS(cache);
	}
	return cache;
}

Uint32 Direct3DRMRTXRenderer::GetMeshId(IDirect3DRMMesh* mesh, const MeshGroup* meshGroup)
{
	auto destroyCb = [](Direct3DRMRenderer* renderer, Uint32 id) {
		auto* self = static_cast<Direct3DRMRTXRenderer*>(renderer);
		auto& c = self->m_meshs[id];
		self->RetireBuffer(c.vertexBuffer, c.vertexMemory);
		self->RetireBuffer(c.indexBuffer, c.indexMemory);
		self->RetireAS(c.blas, c.blasBuffer, c.blasMemory);
		c = {};
	};

	for (Uint32 i = 0; i < m_meshs.size(); ++i) {
		auto& cache = m_meshs[i];
		if (cache.meshGroup == meshGroup) {
			if (cache.version != meshGroup->version) {
				RetireBuffer(cache.vertexBuffer, cache.vertexMemory);
				RetireBuffer(cache.indexBuffer, cache.indexMemory);
				RetireAS(cache.blas, cache.blasBuffer, cache.blasMemory);
				cache = UploadMeshData(*meshGroup, true);
			}
			return i;
		}
	}

	auto newCache = UploadMeshData(*meshGroup, true);
	for (Uint32 i = 0; i < m_meshs.size(); ++i) {
		if (!m_meshs[i].meshGroup) {
			m_meshs[i] = newCache;
			RegisterDestroyCallback(mesh, i, destroyCb);
			return i;
		}
	}
	m_meshs.push_back(newCache);
	RegisterDestroyCallback(mesh, (Uint32) (m_meshs.size() - 1), destroyCb);
	return (Uint32) (m_meshs.size() - 1);
}

// ---------------------------------------------------------------------------
// Frame interface
// ---------------------------------------------------------------------------

void Direct3DRMRTXRenderer::PushLights(const SceneLight* lights, size_t count)
{
	m_lights.assign(lights, lights + std::min(count, (size_t) RTX_MAX_LIGHTS));
}

void Direct3DRMRTXRenderer::SetFrustumPlanes(const Plane* frustumPlanes)
{
}

void Direct3DRMRTXRenderer::SetProjection(const D3DRMMATRIX4D& projection, D3DVALUE front, D3DVALUE back)
{
	m_front = front;
	m_back = back;
	memcpy(m_projection, projection, sizeof(D3DRMMATRIX4D));
}

HRESULT Direct3DRMRTXRenderer::BeginFrame()
{
	m_draws.clear();
	m_transparentDraws.clear();
	m_uiDraws.clear();
	m_transparencyEnabled = false;
	m_gotView = false;
	m_frameDirty = true;
	return DD_OK;
}

void Direct3DRMRTXRenderer::EnableTransparency()
{
	m_transparencyEnabled = true;
}

void Direct3DRMRTXRenderer::SubmitDraw(
	DWORD meshId,
	const D3DRMMATRIX4D& modelViewMatrix,
	const D3DRMMATRIX4D& worldMatrix,
	const D3DRMMATRIX4D& viewMatrix,
	const Matrix3x3& normalMatrix,
	const Appearance& appearance
)
{
	if (meshId >= m_meshs.size() || !m_meshs[meshId].vertexBuffer) {
		return;
	}
	RTXDraw draw = {};
	draw.meshId = meshId;
	memcpy(draw.modelViewMatrix, modelViewMatrix, sizeof(D3DRMMATRIX4D));
	memcpy(draw.worldMatrix, worldMatrix, sizeof(D3DRMMATRIX4D));
	memcpy(draw.normalMatrix, normalMatrix, sizeof(Matrix3x3));
	draw.appearance = appearance;
	if (m_transparencyEnabled) {
		m_transparentDraws.push_back(draw);
	}
	else {
		m_draws.push_back(draw);
	}
	if (!m_gotView) {
		memcpy(m_viewMatrix, viewMatrix, sizeof(D3DRMMATRIX4D));
		m_gotView = true;
	}
}

HRESULT Direct3DRMRTXRenderer::FinalizeFrame()
{
	m_frameDirty = true;
	return DD_OK;
}

void Direct3DRMRTXRenderer::Clear(float r, float g, float b)
{
	m_clearColor[0] = r;
	m_clearColor[1] = g;
	m_clearColor[2] = b;
	m_clearColor[3] = 1.f;
	m_frameDirty = true;
}

void Direct3DRMRTXRenderer::Resize(int width, int height, const ViewportTransform& viewportTransform)
{
	m_width = width;
	m_height = height;
	m_viewportTransform = viewportTransform;
	if (!m_device) {
		return;
	}
	vkDeviceWaitIdle(m_device);
	for (auto& slot : m_slots) {
		slot.submitted = false;
	}
	ReapPending();
	DestroyTargets();
	InitTargets();
	if (m_postSet) {
		VkDescriptorImageInfo ii = {m_sampler, m_hdrView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		w.dstSet = m_postSet;
		w.dstBinding = 0;
		w.descriptorCount = 1;
		w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		w.pImageInfo = &ii;
		vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
	}
	m_frameDirty = true;
}

void Direct3DRMRTXRenderer::SetDither(bool dither)
{
	m_dither = dither;
}

// ---------------------------------------------------------------------------
// Frame recording
// ---------------------------------------------------------------------------

void Direct3DRMRTXRenderer::WriteFrameUBO(RTXFrameSlot& slot)
{
	RTXFrameUBO ubo = {};
	memcpy(ubo.proj, m_projection, sizeof(m_projection));
	memcpy(ubo.view, m_viewMatrix, sizeof(m_viewMatrix));

	// Camera world position for a rigid row-vector view transform [R|t]:
	// v_view = v_world * R + t  =>  camera world = -t * R^-1 = -t * R^T.
	float tx = m_viewMatrix[3][0], ty = m_viewMatrix[3][1], tz = m_viewMatrix[3][2];
	ubo.cameraPos[0] = -(tx * m_viewMatrix[0][0] + ty * m_viewMatrix[0][1] + tz * m_viewMatrix[0][2]);
	ubo.cameraPos[1] = -(tx * m_viewMatrix[1][0] + ty * m_viewMatrix[1][1] + tz * m_viewMatrix[1][2]);
	ubo.cameraPos[2] = -(tx * m_viewMatrix[2][0] + ty * m_viewMatrix[2][1] + tz * m_viewMatrix[2][2]);
	ubo.cameraPos[3] = 1.0f;

	ubo.ambient[0] = m_clearColor[0];
	ubo.ambient[1] = m_clearColor[1];
	ubo.ambient[2] = m_clearColor[2];
	ubo.ambient[3] = 0.25f;

	ubo.rtParams[0] = (float) (m_rtEnabled ? m_rtFlags : 0);
	ubo.rtParams[1] = (float) m_lights.size();
	ubo.rtParams[2] = m_front;
	ubo.rtParams[3] = m_back;

	for (size_t i = 0; i < m_lights.size(); ++i) {
		ubo.lights[i] = m_lights[i];
	}
	memcpy(slot.frameUBOMap, &ubo, sizeof(ubo));
}

VkDescriptorSet Direct3DRMRTXRenderer::AllocDrawSet(
	RTXFrameSlot& slot,
	uint32_t uboOffset,
	VkImageView texView,
	VkSampler sampler
)
{
	VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	ai.descriptorPool = slot.drawDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts = &m_set1;
	VkDescriptorSet set = VK_NULL_HANDLE;
	if (vkAllocateDescriptorSets(m_device, &ai, &set) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}
	VkDescriptorBufferInfo bi = {slot.drawUBO, uboOffset, RTX_DRAW_UBO_STRIDE};
	VkDescriptorImageInfo ii = {sampler, texView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
	VkWriteDescriptorSet w[2] = {};
	w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	w[0].dstSet = set;
	w[0].dstBinding = 0;
	w[0].descriptorCount = 1;
	w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	w[0].pBufferInfo = &bi;
	w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	w[1].dstSet = set;
	w[1].dstBinding = 1;
	w[1].descriptorCount = 1;
	w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	w[1].pImageInfo = &ii;
	vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
	return set;
}

static void RTXWriteDrawUBO(char* base, uint32_t slot, const RTXDraw& draw)
{
	RTXDrawUBO ubo = {};
	memcpy(ubo.modelView, draw.modelViewMatrix, sizeof(ubo.modelView));
	// The 3x3 normal matrix goes into the top-left of a mat4, uploaded raw:
	// GLSL reads it column-major which yields the correct normal transform.
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			ubo.normal[i * 4 + j] = draw.normalMatrix[i][j];
		}
	}
	memcpy(ubo.world, draw.worldMatrix, sizeof(ubo.world));
	ubo.color[0] = draw.appearance.color.r / 255.0f;
	ubo.color[1] = draw.appearance.color.g / 255.0f;
	ubo.color[2] = draw.appearance.color.b / 255.0f;
	ubo.color[3] = draw.appearance.color.a / 255.0f;
	ubo.params[0] = draw.appearance.shininess;
	ubo.params[1] = (draw.appearance.textureId != NO_TEXTURE_ID) ? 1.f : 0.f;
	ubo.params[2] = draw.appearance.flat ? 1.f : 0.f;
	memcpy(base + slot * RTX_DRAW_UBO_STRIDE, &ubo, sizeof(ubo));
}

bool Direct3DRMRTXRenderer::RecordFrame()
{
	RTXFrameSlot& slot = m_slots[m_slot];
	WaitForSlot(slot);
	ReapPending();
	vkResetDescriptorPool(m_device, slot.drawDescPool, 0);
	vkResetCommandBuffer(slot.cmd, 0);

	WriteFrameUBO(slot);

	size_t totalDraws = m_draws.size() + m_transparentDraws.size() + m_uiDraws.size();
	if (totalDraws > RTX_MAX_DRAWS) {
		SDL_LogError(LOG_CATEGORY_MINIWIN, "Too many draws in a frame (%u)", (unsigned) totalDraws);
		m_draws.resize(RTX_MAX_DRAWS);
		m_transparentDraws.clear();
		m_uiDraws.clear();
	}

	char* uboBase = (char*) slot.drawUBOMap;
	uint32_t uboSlot = 0;
	std::vector<uint32_t> opaqueSlots, transSlots;
	for (auto& d : m_draws) {
		RTXWriteDrawUBO(uboBase, uboSlot, d);
		opaqueSlots.push_back(uboSlot++);
	}
	for (auto& d : m_transparentDraws) {
		RTXWriteDrawUBO(uboBase, uboSlot, d);
		transSlots.push_back(uboSlot++);
	}
	for (auto& d : m_uiDraws) {
		RTXDrawUBO ubo = {};
		memcpy(ubo.modelView, d.transform, sizeof(ubo.modelView));
		static const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		memcpy(ubo.normal, identity, sizeof(identity));
		memcpy(ubo.world, identity, sizeof(identity));
		ubo.color[0] = d.color.r;
		ubo.color[1] = d.color.g;
		ubo.color[2] = d.color.b;
		ubo.color[3] = d.color.a;
		ubo.params[1] = 1.f;
		memcpy(uboBase + uboSlot * RTX_DRAW_UBO_STRIDE, &ubo, sizeof(ubo));
		d.drawUBOSlot = uboSlot++;
	}

	if (m_rtEnabled && (m_draws.size() + m_transparentDraws.size())) {
		size_t instCount = m_draws.size() + m_transparentDraws.size();
		VkDeviceSize need = instCount * sizeof(RTXInstanceInfo);
		if (slot.instSSBOSize < need) {
			if (slot.instSSBO) {
				RetireBuffer(slot.instSSBO, slot.instSSBOMemory);
				slot.instSSBOMap = nullptr;
			}
			if (!CreateBuffer(
					need * 2,
					VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					slot.instSSBO,
					slot.instSSBOMemory,
					true,
					&slot.instSSBOMap
				)) {
				return false;
			}
			slot.instSSBOSize = need * 2;
		}
		auto* info = (RTXInstanceInfo*) slot.instSSBOMap;
		size_t i = 0;
		auto fillInst = [&](const std::vector<RTXDraw>& draws) {
			for (const auto& d : draws) {
				auto& dst = info[i++];
				const RTXMeshCache& mesh = m_meshs[d.meshId];
				VkDeviceAddress va = BufferAddress(mesh.vertexBuffer);
				VkDeviceAddress ia = BufferAddress(mesh.indexBuffer);
				dst.vbLo = (uint32_t) (va & 0xFFFFFFFFull);
				dst.vbHi = (uint32_t) (va >> 32);
				dst.ibLo = (uint32_t) (ia & 0xFFFFFFFFull);
				dst.ibHi = (uint32_t) (ia >> 32);
				dst.color[0] = d.appearance.color.r / 255.f;
				dst.color[1] = d.appearance.color.g / 255.f;
				dst.color[2] = d.appearance.color.b / 255.f;
				dst.color[3] = d.appearance.color.a / 255.f;
				dst.indexCount = mesh.indexCount;
				dst.flags = 0;
				dst.textureId = d.appearance.textureId;
			}
		};
		fillInst(m_draws);
		fillInst(m_transparentDraws);
	}

	VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	vkBeginCommandBuffer(slot.cmd, &bi);

	if (m_rtEnabled) {
		if (!BuildTLAS(slot, slot.cmd)) {
			vkEndCommandBuffer(slot.cmd);
			return false;
		}
	}
	UpdateSet0(slot);

	VkViewport vp = {0.f, (float) m_height, (float) m_width, -(float) m_height, 0.f, 1.f};
	VkRect2D scissor = {{0, 0}, {(uint32_t) m_width, (uint32_t) m_height}};

	{
		VkRenderPassBeginInfo rp = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rp.renderPass = m_hdrPass;
		rp.framebuffer = m_hdrFB;
		rp.renderArea = scissor;
		VkClearValue clearVals[2] = {};
		clearVals[0].color = {{m_clearColor[0], m_clearColor[1], m_clearColor[2], 1.f}};
		clearVals[1].depthStencil = {1.f, 0};
		rp.clearValueCount = 2;
		rp.pClearValues = clearVals;
		vkCmdBeginRenderPass(slot.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(slot.cmd, 0, 1, &vp);
		vkCmdSetScissor(slot.cmd, 0, 1, &scissor);

		auto record = [&](const std::vector<RTXDraw>& draws, const std::vector<uint32_t>& slots, VkPipeline pipe) {
			vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
			vkCmdBindDescriptorSets(
				slot.cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				m_sceneLayout,
				0,
				1,
				&slot.set0,
				0,
				nullptr
			);
			for (size_t i = 0; i < draws.size(); ++i) {
				const RTXDraw& d = draws[i];
				const RTXMeshCache& mesh = m_meshs[d.meshId];
				VkImageView tv = (d.appearance.textureId != NO_TEXTURE_ID &&
								  d.appearance.textureId < m_textures.size() &&
								  m_textures[d.appearance.textureId].view)
									 ? m_textures[d.appearance.textureId].view
									 : m_dummyView;
				VkDescriptorSet ds = AllocDrawSet(slot, slots[i] * RTX_DRAW_UBO_STRIDE, tv, m_sampler);
				if (!ds) {
					continue;
				}
				vkCmdBindDescriptorSets(
					slot.cmd,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					m_sceneLayout,
					1,
					1,
					&ds,
					0,
					nullptr
				);
				VkDeviceSize off = 0;
				vkCmdBindVertexBuffers(slot.cmd, 0, 1, &mesh.vertexBuffer, &off);
				vkCmdBindIndexBuffer(slot.cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT16);
				vkCmdDrawIndexed(slot.cmd, mesh.indexCount, 1, 0, 0, 0);
			}
		};
		record(m_draws, opaqueSlots, m_opaquePipeline);
		record(m_transparentDraws, transSlots, m_transparentPipeline);
		vkCmdEndRenderPass(slot.cmd);
	}

	{
		VkRenderPassBeginInfo rp = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rp.renderPass = m_ldrPass;
		rp.framebuffer = m_ldrFB;
		rp.renderArea = scissor;
		VkClearValue clearVal = {};
		rp.clearValueCount = 1;
		rp.pClearValues = &clearVal;
		vkCmdBeginRenderPass(slot.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(slot.cmd, 0, 1, &vp);
		vkCmdSetScissor(slot.cmd, 0, 1, &scissor);

		vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
		vkCmdBindDescriptorSets(
			slot.cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_postLayout,
			0,
			1,
			&m_postSet,
			0,
			nullptr
		);
		float postParams[4] = {1.0f, 0.25f, 1.0f, m_dither ? 1.f : 0.f};
		vkCmdPushConstants(slot.cmd, m_postLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, postParams);
		vkCmdDraw(slot.cmd, 3, 1, 0, 0);

		if (!m_uiDraws.empty()) {
			vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_uiPipeline);
			vkCmdBindDescriptorSets(
				slot.cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				m_sceneLayout,
				0,
				1,
				&slot.set0,
				0,
				nullptr
			);
			for (auto& d : m_uiDraws) {
				VkImageView tv = (d.textureId != NO_TEXTURE_ID && d.textureId < m_textures.size() &&
								  m_textures[d.textureId].view)
									 ? m_textures[d.textureId].view
									 : m_dummyView;
				VkDescriptorSet ds =
					AllocDrawSet(slot, d.drawUBOSlot * RTX_DRAW_UBO_STRIDE, tv, m_uiSampler);
				if (!ds) {
					continue;
				}
				vkCmdBindDescriptorSets(
					slot.cmd,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					m_sceneLayout,
					1,
					1,
					&ds,
					0,
					nullptr
				);
				VkRect2D sc = {{d.scissor.x, d.scissor.y}, {(uint32_t) d.scissor.w, (uint32_t) d.scissor.h}};
				vkCmdSetScissor(slot.cmd, 0, 1, &sc);
				VkDeviceSize off = 0;
				vkCmdBindVertexBuffers(slot.cmd, 0, 1, &m_uiMeshCache.vertexBuffer, &off);
				vkCmdBindIndexBuffer(slot.cmd, m_uiMeshCache.indexBuffer, 0, VK_INDEX_TYPE_UINT16);
				vkCmdDrawIndexed(slot.cmd, m_uiMeshCache.indexCount, 1, 0, 0, 0);
			}
			vkCmdSetScissor(slot.cmd, 0, 1, &scissor);
		}
		vkCmdEndRenderPass(slot.cmd);
	}

	// Present: acquire the swapchain image and blit the LDR target into it.
	bool acquired = false;
	if (m_surface && m_swapchain) {
		VkResult res =
			vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, m_acquireSem, VK_NULL_HANDLE, &m_presentImage);
		if (res == VK_ERROR_OUT_OF_DATE_KHR) {
			vkEndCommandBuffer(slot.cmd);
			vkDeviceWaitIdle(m_device);
			DestroySwapchain();
			InitSwapchain();
			return true;
		}
		acquired = (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR);
		if (acquired) {
			VkImageMemoryBarrier pre = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
			pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			pre.image = m_swapImages[m_presentImage];
			pre.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vkCmdPipelineBarrier(
				slot.cmd,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0,
				0,
				nullptr,
				0,
				nullptr,
				1,
				&pre
			);
			VkImageBlit blit = {};
			blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			blit.srcOffsets[1] = {(int32_t) m_width, (int32_t) m_height, 1};
			blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			blit.dstOffsets[1] = {(int32_t) m_swapExtent.width, (int32_t) m_swapExtent.height, 1};
			vkCmdBlitImage(
				slot.cmd,
				m_ldrImage,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				m_swapImages[m_presentImage],
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&blit,
				VK_FILTER_NEAREST
			);
			VkImageMemoryBarrier post = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
			post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			post.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			post.image = m_swapImages[m_presentImage];
			post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			vkCmdPipelineBarrier(
				slot.cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0,
				nullptr,
				0,
				nullptr,
				1,
				&post
			);
		}
	}

	vkEndCommandBuffer(slot.cmd);
	vkResetFences(m_device, 1, &slot.fence);

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &slot.cmd;
	if (acquired) {
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &m_acquireSem;
		si.pWaitDstStageMask = &waitStage;
	}
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores = &slot.renderDone;
	if (vkQueueSubmit(m_queue, 1, &si, slot.fence) != VK_SUCCESS) {
		return false;
	}
	slot.submitted = true;
	m_presentPending = acquired;
	m_frameDirty = false;
	m_slot = (m_slot + 1) % RTX_FRAMES_IN_FLIGHT;
	return true;
}

void Direct3DRMRTXRenderer::Flip()
{
	if (m_frameDirty) {
		if (!RecordFrame()) {
			return;
		}
	}
	if (m_presentPending && m_surface && m_swapchain) {
		uint32_t prevSlot = (m_slot + RTX_FRAMES_IN_FLIGHT - 1) % RTX_FRAMES_IN_FLIGHT;
		VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &m_slots[prevSlot].renderDone;
		pi.swapchainCount = 1;
		pi.pSwapchains = &m_swapchain;
		pi.pImageIndices = &m_presentImage;
		VkResult res = vkQueuePresentKHR(m_queue, &pi);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
			vkDeviceWaitIdle(m_device);
			for (auto& slot : m_slots) {
				slot.submitted = false;
			}
			DestroySwapchain();
			InitSwapchain();
		}
	}
	m_presentPending = false;
	ReapPending();
}

void Direct3DRMRTXRenderer::Draw2DImage(
	Uint32 textureId,
	const SDL_Rect& srcRect,
	const SDL_Rect& dstRect,
	FColor color
)
{
	RTXUIDraw draw = {};
	draw.textureId = textureId;
	draw.color = color;

	SDL_Rect expandedDstRect = dstRect;
	if (textureId != NO_TEXTURE_ID && textureId < m_textures.size() &&
		m_textures[textureId].texture) {
		auto surface = static_cast<DirectDrawSurfaceImpl*>(m_textures[textureId].texture->m_surface);
		float scaleX = static_cast<float>(dstRect.w) / srcRect.w;
		float scaleY = static_cast<float>(dstRect.h) / srcRect.h;
		expandedDstRect = {
			static_cast<int>(std::round(dstRect.x - srcRect.x * scaleX)),
			static_cast<int>(std::round(dstRect.y - srcRect.y * scaleY)),
			static_cast<int>(std::round(static_cast<float>(surface->m_surface->w) * scaleX)),
			static_cast<int>(std::round(static_cast<float>(surface->m_surface->h) * scaleY)),
		};
	}

	D3DRMMATRIX4D t2d, ortho;
	Create2DTransformMatrix(
		expandedDstRect,
		m_viewportTransform.scale,
		m_viewportTransform.offsetX,
		m_viewportTransform.offsetY,
		t2d
	);
	CreateOrthographicProjection((float) m_width, (float) m_height, ortho);
	MultiplyMatrix(draw.transform, t2d, ortho);

	draw.scissor.x =
		static_cast<int>(std::round(dstRect.x * m_viewportTransform.scale + m_viewportTransform.offsetX));
	draw.scissor.y =
		static_cast<int>(std::round(dstRect.y * m_viewportTransform.scale + m_viewportTransform.offsetY));
	draw.scissor.w = static_cast<int>(std::round(dstRect.w * m_viewportTransform.scale));
	draw.scissor.h = static_cast<int>(std::round(dstRect.h * m_viewportTransform.scale));

	m_uiDraws.push_back(draw);
	m_frameDirty = true;
}

void Direct3DRMRTXRenderer::Download(SDL_Surface* target)
{
	if (!m_device) {
		return;
	}
	vkQueueWaitIdle(m_queue);

	const int offsetX = static_cast<int>(m_viewportTransform.offsetX);
	const int offsetY = static_cast<int>(m_viewportTransform.offsetY);
	const int width = m_width - offsetX * 2;
	const int height = m_height - offsetY * 2;
	if (width <= 0 || height <= 0) {
		return;
	}

	VkBuffer staging;
	VkDeviceMemory stagingMem;
	void* mapped = nullptr;
	if (!CreateBuffer(
			(VkDeviceSize) width * height * 4,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			staging,
			stagingMem,
			true,
			&mapped
		)) {
		return;
	}

	VkCommandBuffer cmd = BeginImmediate();
	VkBufferImageCopy copy = {};
	copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy.imageOffset = {offsetX, offsetY, 0};
	copy.imageExtent = {(uint32_t) width, (uint32_t) height, 1};
	vkCmdCopyImageToBuffer(cmd, m_ldrImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &copy);
	vkEndCommandBuffer(cmd);
	VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(m_queue);
	vkFreeCommandBuffers(m_device, m_immediatePool, 1, &cmd);

	SDL_PixelFormat fmt =
		(m_swapFormat == VK_FORMAT_B8G8R8A8_UNORM) ? SDL_PIXELFORMAT_BGRA32 : SDL_PIXELFORMAT_RGBA32;
	SDL_Surface* renderedImage = SDL_CreateSurfaceFrom(width, height, fmt, mapped, width * 4);
	if (renderedImage) {
		SDL_BlitSurfaceScaled(renderedImage, nullptr, target, nullptr, SDL_SCALEMODE_NEAREST);
		SDL_DestroySurface(renderedImage);
	}
	vkUnmapMemory(m_device, stagingMem);
	vkDestroyBuffer(m_device, staging, nullptr);
	vkFreeMemory(m_device, stagingMem, nullptr);
}
