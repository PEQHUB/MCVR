#include "com_radiance_client_pipeline_Pipeline.h"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <iostream>

JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_buildNative(JNIEnv *, jclass, jlong paramsLongPtr) {
    WorldPipelineBuildParams *params = reinterpret_cast<WorldPipelineBuildParams *>(paramsLongPtr);
    auto pipeline = Renderer::instance().framework()->pipeline();
    if (pipeline != nullptr) Renderer::instance().framework()->pipeline()->buildWorldPipelineBlueprint(params);
}

JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_collectNativeModules(JNIEnv *, jclass) {
    Pipeline::collectWorldModules();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_pipeline_Pipeline_recollectNativeModules(JNIEnv *, jclass) {
    Pipeline::recollectWorldModules();
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_pipeline_Pipeline_isNativeModuleAvailable(JNIEnv *env,
                                                                                              jclass,
                                                                                              jstring name) {
    if (name == nullptr) return JNI_FALSE;
    const char *nativeString = env->GetStringUTFChars(name, nullptr);
    if (nativeString == nullptr) return JNI_FALSE;
    bool available = Pipeline::worldModuleConstructors.find(nativeString) != Pipeline::worldModuleConstructors.end();
    env->ReleaseStringUTFChars(name, nativeString);
    return available ? JNI_TRUE : JNI_FALSE;
}
