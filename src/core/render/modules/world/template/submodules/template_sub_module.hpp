#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

class TemplateModule;
struct TemplateModuleContext;

class TemplateSub : public SharedObject<TemplateSub> {
    public:
    TemplateSub();
     
    void init(std::shared_ptr<TemplateModule>);

    private:
    std::weak_ptr<TemplateModule> templateModule_;
};

struct TemplateSubContext : public SharedObject<TemplateSubContext> {
    std::weak_ptr<TemplateModuleContext> templateModuleContext;
};