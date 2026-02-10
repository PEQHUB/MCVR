#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <condition_variable>
#include <future>
#include <thread>

class Framework;
class Chunks;
class Entities;

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

    void close();

  private:
    std::shared_ptr<Chunks> chunks_;
    std::shared_ptr<Entities> entities_;

    glm::dvec3 cameraPos_ = {0, 0, 0};

    bool shouldRenderWorld_;
};