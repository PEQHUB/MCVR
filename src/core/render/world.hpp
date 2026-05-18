#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <condition_variable>
#include <future>
#include <memory>
#include <thread>

class Framework;
class Chunks;
class Entities;
#include "core/render/extended_chunk_manager.hpp"

class World : public SharedObject<World> {
  public:
    enum VertexFormats {
        POSITION_COLOR_TEXTURE_LIGHT_NORMAL,
        POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL,
        POSITION_TEXTURE_COLOR_LIGHT,
        POSITION,
        POSITION_COLOR,
        LINES,
        POSITION_COLOR_LIGHT,
        POSITION_TEXTURE,
        POSITION_TEXTURE_COLOR,
        POSITION_COLOR_TEXTURE_LIGHT,
        POSITION_TEXTURE_LIGHT_COLOR,
        POSITION_TEXTURE_COLOR_NORMAL,
        PBR_TRIANGLE,
        NUM_VERTEX_FORMATS,
    };

    enum GeometryTypes {
        SHADOW,
        WORLD_SOLID,
        WORLD_TRANSPARENT,
        WORLD_NO_REFLECT,
        WORLD_CLOUD,
        BOAT_WATER_MASK,
        END_PORTAL,
        END_GATE_WAY,
        WORLD_DISPLACED,      // DDA intersection shader — procedural AABB geometry
        NUM_GEOMETRY_TYPES,
    };

    enum Coordinates {
        WORLD,
        CAMERA,
        CAMERA_SHIFT,
    };

    enum class DrawMode {
        LINES,
        LINE_STRIP,
        DEBUG_LINES,
        DEBUG_LINE_STRIP,
        TRIANGLES,
        TRIANGLE_STRIP,
        TRIANGLE_FAN,
        QUADS,
    };

    World(std::shared_ptr<Framework> framework);

    void resetFrame();
    bool &shouldRender();

    std::shared_ptr<Chunks> chunks();
    std::shared_ptr<Entities> entities();

    void setCameraPos(glm::dvec3 cameraPos);
    glm::dvec3 getCameraPos();

    /// Start extended chunk loading if configured. Called after world init.
    void startExtendedChunkLoading(uint32_t javaRenderDistance, uint32_t javaChunkCount);
    ExtendedChunkManager* extendedChunkManager() { return extendedChunkMgr_.get(); }

    void close();

  private:
    std::shared_ptr<Chunks> chunks_;
    std::shared_ptr<Entities> entities_;
    std::unique_ptr<ExtendedChunkManager> extendedChunkMgr_;

    glm::dvec3 cameraPos_ = {0, 0, 0};

    bool shouldRenderWorld_;
};