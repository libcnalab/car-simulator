#include "CarSim/Render/WorldRenderer.hpp"

#include "CarSim/Render/GroundShadowBaker.hpp"

#include "CarSim/Render/Image.hpp"
#include "CarSim/Render/ProceduralTextures.hpp"

#include "CnaEngine/Graphics/Camera.hpp"
#include "CnaEngine/Graphics/ForwardRenderer.hpp"
#include "CnaEngine/Graphics/Image.hpp"
#include "CnaEngine/Graphics/Material.hpp"
#include "CnaEngine/Graphics/RenderQueue.hpp"
#include "CnaEngine/Graphics/RenderStatistics.hpp"
#include "CnaEngine/Graphics/Visibility.hpp"

#include "Microsoft/Xna/Framework/Graphics/BlendState.hpp"
#include "Microsoft/Xna/Framework/Graphics/DepthStencilState.hpp"
#include "Microsoft/Xna/Framework/Graphics/EffectPass.hpp"
#include "Microsoft/Xna/Framework/Graphics/EffectPassCollection.hpp"
#include "Microsoft/Xna/Framework/Graphics/EffectTechnique.hpp"
#include "Microsoft/Xna/Framework/Graphics/SamplerState.hpp"
#include "Microsoft/Xna/Framework/Graphics/SamplerStateCollection.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>
#include <chrono>
#include <limits>
#include <thread>
#include <cmath>

namespace CarSim::Render
{
    using namespace Microsoft::Xna::Framework;
    using namespace Microsoft::Xna::Framework::Graphics;

    namespace Engine = CnaEngine::Graphics;

    namespace
    {
        void ApplyAll(Effect& effect, GraphicsDevice& device, const GpuMesh& mesh)
        {
            auto& passes = effect.getCurrentTechniqueProperty()->getPassesProperty();
            for (int i = 0; i < passes.getCountProperty(); ++i) {
                passes[i]->Apply();
                mesh.Draw(device);
            }
        }

        constexpr float kAnyDistance = std::numeric_limits<float>::infinity();
        // AlphaTestEffect's Greater 110: the engine keeps alpha >= cutoff.
        constexpr float kCardCutoff = 111.0f / 255.0f;
        constexpr int kSurfaceCount = 7;

        /// The engine's culling: bounds entirely beyond `range` or outside the frustum are dropped.
        bool Visible(const Engine::Camera& camera, const BoundingSphere& bounds, const float range = kAnyDistance)
        {
            Engine::VisibilitySettings settings;
            settings.maxDistance = range;
            return Engine::Classify(bounds, 1, camera, settings) == Engine::Visibility::Visible;
        }

        /// `order` < another's draws first within its bucket (road surfaces before markings).
        void Submit(Engine::RenderQueue& queue, const GpuMesh& mesh, const Engine::Material& material,
                    const BoundingSphere& bounds, const std::int32_t order = 0)
        {
            Engine::RenderItem item;
            item.mesh = &mesh.Engine();
            item.material = &material;
            item.bounds = bounds;
            item.order = order;
            queue.Submit(item);
        }

        /// Draws and empties one segment of the frame.
        void Render(Engine::ForwardRenderer& renderer, Engine::RenderQueue& queue, const Engine::Camera& camera,
                    WorldRenderStats& stats)
        {
            if (queue.Count() == 0) {
                return;
            }
            queue.Sort(camera.Position());
            Engine::RenderStatistics drawn;
            renderer.Render(queue, camera, &drawn);
            queue.Clear();
            stats.drawCalls += static_cast<int>(drawn.drawCalls);
            stats.triangles += static_cast<int>(drawn.primitives);
            GpuMesh::RecordSubmissions(static_cast<long long>(drawn.drawCalls), static_cast<long long>(drawn.primitives));
        }

        /// LightingRig::Apply(BasicEffect&) as the engine's scene lighting.
        void SetLighting(Engine::ForwardLighting& lighting, const LightingRig& rig)
        {
            lighting.defaultLighting = false;
            lighting.lightDirection = rig.sunDirection;
            lighting.lightColor = rig.sunColor;
            lighting.fillLights = {{Vector3(0.0f, -1.0f, 0.0f), rig.skyFillColor, Vector3(0.0f, 0.0f, 0.0f)},
                                   {Vector3(0.3f, 1.0f, 0.2f), rig.groundBounceColor, Vector3(0.0f, 0.0f, 0.0f)}};
            lighting.ambientColor = rig.skyAmbient;
            lighting.perPixelLighting = true;
            lighting.fogEnabled = true;
            lighting.fogColor = rig.fogColor;
            lighting.fogStart = rig.fogStart;
            lighting.fogEnd = rig.fogEnd;
        }
    }

    struct WorldRenderer::EnginePasses
    {
        EnginePasses(GraphicsDevice& device, const float lod1, const float lod2)
            : renderer(device), beamObjects(device), beamCards(device), terrainLod({lod1, lod2}, 0.0f)
        {
        }

        Engine::ForwardRenderer renderer;
        // The headlight fill is a second light setup over what is already drawn: renderers of
        // its own, their draws added on top.
        Engine::ForwardRenderer beamObjects;
        Engine::ForwardRenderer beamCards;
        Engine::Camera camera;
        Engine::RenderQueue queue;
        Engine::RenderQueue beamQueue;
        Engine::LodSelector terrainLod;

        Engine::Material terrain;
        std::array<Engine::Material, kSurfaceCount> surfaces;   // by Surface
        std::vector<Engine::Material> objects;                   // parallel to objectBatches_
        std::vector<Engine::Material> trees;                     // parallel to treeBatches_
        std::vector<Engine::Material> signs;                     // parallel to signBatches_
        std::vector<Engine::Material> litObjects;                // the headlight fill, parallel again
        std::vector<Engine::Material> litTrees;
        std::vector<Engine::Material> litSigns;
        Engine::Material lamp;
    };

    struct WorldRenderer::ShadowBakeJob
    {
        struct MacroLevel
        {
            int width = 0;
            int height = 0;
            std::vector<std::uint8_t> rgba;
        };
        Vector3 sun;
        std::vector<MacroLevel> macroLevels;
        std::vector<std::vector<Color>> colours;   // per road batch, in batch order
        std::atomic<bool> done{false};
        std::thread worker;
        double seconds = 0.0;
    };

    WorldRenderer::WorldRenderer(GraphicsDevice& device, const LightingRig& rig, const Map::MapWorld& world, const BitmapFont* signFont)
        : world_(world), rig_(rig)
    {
        grass_ = UploadTexture(device, Textures::Grass(512, 1u), true);
        asphalt_ = UploadTexture(device, Textures::Asphalt(512, 2u), true);
        gravel_ = UploadTexture(device, Textures::Gravel(256, 4u), true);
        paving_ = UploadTexture(device, Textures::PavingSlabs(256, 5u), true);
        cobbles_ = UploadTexture(device, Textures::Cobbles(256, 9u), true);
        concrete_ = UploadTexture(device, Textures::Plaster(128, Rgb::FromBytes(176, 174, 168), 6u), true);
        marking_ = UploadTexture(device, Textures::MarkingPaint(64, 7u), true);
        white_ = UploadTexture(device, Textures::Solid(4, Color(255, 255, 255, 255)), false);

        // Ground shadows of buildings and trees at two texels per terrain cell (capped), baked
        // into the terrain macro and the road vertex colours.
        const auto& terrain = world_.Terrain();
        const int macroW = std::min(2048, std::max(1, 2 * terrain.Columns()));
        const int macroH = std::min(2048, std::max(1, 2 * terrain.Rows()));
        const Image shadow = GroundShadows::Bake(world_, bakeRig_.sunDirection, macroW, macroH);
        shadowSun_ = bakeRig_.sunDirection;
        Image tint(macroW, macroH);
        BuildMacroTexture(device, shadow, tint);
        BuildTerrain(device);
        BuildRoads(device, shadow, tint);
        BuildIntersections(device, shadow);
        BuildPavedAreas(device, shadow);
        BuildObjects(device);
        BuildLampLights(device);
        BuildTrees(device);
        BuildSigns(device, signFont);
        BuildEngineMaterials(device);

        // Wet-road sheen. Wet asphalt is dark under your wheels and a mirror at the far end of
        // the street, because the sky's reflection climbs with the grazing angle -- and down a
        // road, distance is the grazing angle. A BasicEffect's fog is a distance ramp, so a
        // second additive pass over the road with a black diffuse and the sky as the fog colour
        // puts exactly that sheen on it: nothing near, sky far. No renderer internals, no custom
        // shader, and it costs one extra pass over the road batches only while the road is wet.
        puddleMask_ = UploadTexture(device, Textures::PuddleMask(256, 71u), true);
        snowTexture_ = UploadTexture(device, Textures::SnowLayer(256, 29u), true);
        snowEffect_ = std::make_unique<BasicEffect>(device);
        snowEffect_->setLightingEnabledProperty(false);
        snowEffect_->setTextureEnabledProperty(true);
        snowEffect_->setVertexColorEnabledProperty(false);
        snowEffect_->setFogEnabledProperty(true);
        puddleEffect_ = std::make_unique<BasicEffect>(device);
        puddleEffect_->setLightingEnabledProperty(false);
        puddleEffect_->setTextureEnabledProperty(true);
        puddleEffect_->setVertexColorEnabledProperty(false);
        puddleEffect_->setFogEnabledProperty(true);
        puddleEffect_->setFogColorProperty(Vector3(0.0f, 0.0f, 0.0f));   // additive: fog fades it out
        roadSheenEffect_ = std::make_unique<BasicEffect>(device);
        roadSheenEffect_->setLightingEnabledProperty(false);
        roadSheenEffect_->setTextureEnabledProperty(false);
        roadSheenEffect_->setVertexColorEnabledProperty(false);
        roadSheenEffect_->setDiffuseColorProperty(Vector3(0.0f, 0.0f, 0.0f));
        roadSheenEffect_->setFogEnabledProperty(true);

        markingState_ = std::make_unique<RasterizerState>();
        markingState_->setCullModeProperty(CullMode::CullCounterClockwiseFace);
        markingState_->setDepthBiasProperty(-0.00002f);
        markingState_->setSlopeScaleDepthBiasProperty(-1.0f);
        markingStateMirrored_ = std::make_unique<RasterizerState>();
        markingStateMirrored_->setCullModeProperty(CullMode::CullClockwiseFace);
        markingStateMirrored_->setDepthBiasProperty(-0.00002f);
        markingStateMirrored_->setSlopeScaleDepthBiasProperty(-1.0f);
    }

    void WorldRenderer::BuildEngineMaterials(GraphicsDevice& device)
    {
        engine_ = std::make_unique<EnginePasses>(device, lod1DistanceM, lod2DistanceM);
        EnginePasses& e = *engine_;
        SetLighting(e.renderer.Lighting(), rig_);

        // Terrain: grass detail times the baked macro (DualTextureEffect).
        e.terrain.name = "terrain";
        e.terrain.shading = Engine::ShadingModel::Unlit;
        e.terrain.baseTexture = grass_.get();
        e.terrain.secondaryTexture = macro_.get();
        e.terrain.sampler = &SamplerState::AnisotropicWrap;
        e.terrain.secondarySampler = &SamplerState::LinearClamp;

        // Roads carry their lighting, wear and ground shadows in the vertex colours; markings are
        // pulled towards the camera so they win the depth test against the road under them.
        for (int i = 0; i < kSurfaceCount; ++i) {
            const auto surface = static_cast<Surface>(i);
            Engine::Material& m = e.surfaces[static_cast<std::size_t>(i)];
            m.name = surface == Surface::Marking ? "marking" : "road";
            m.shading = Engine::ShadingModel::Unlit;
            m.vertexColors = true;
            m.baseTexture = TextureFor(surface);
            m.sampler = &SamplerState::AnisotropicWrap;
            if (surface == Surface::Marking) {
                m.depthBias = -0.00002f;
                m.slopeScaleDepthBias = -1.0f;
            }
        }

        // Buildings, props, trunks: lit and textured.
        e.objects.resize(objectBatches_.size());
        e.litObjects.resize(objectBatches_.size());
        for (std::size_t i = 0; i < objectBatches_.size(); ++i) {
            Engine::Material& m = e.objects[i];
            m.name = "object";
            m.baseTexture = objectBatches_[i].texture;
            m.sampler = &SamplerState::AnisotropicWrap;
            Engine::Material& lit = e.litObjects[i];
            lit.name = "headlit object";
            lit.blend = Engine::BlendMode::Additive;
            lit.baseTexture = objectBatches_[i].texture;
            lit.sampler = &SamplerState::AnisotropicWrap;
            lit.diffuseColor = objectBatches_[i].diffuse;
        }

        // Tree and sign cards: alpha-tested, unlit (the light is baked into the cards); signs
        // show both faces. The headlight fill adds them again, still cut out.
        const auto card = [](const char* name, Texture2D* texture, const bool doubleSided) {
            Engine::Material m;
            m.name = name;
            m.shading = Engine::ShadingModel::Unlit;
            m.blend = Engine::BlendMode::AlphaTest;
            m.alphaCutoff = kCardCutoff;
            m.vertexColors = true;
            m.doubleSided = doubleSided;
            m.baseTexture = texture;
            m.sampler = &SamplerState::LinearClamp;
            return m;
        };
        const auto litCard = [&](Texture2D* texture) {
            Engine::Material m = card("headlit card", texture, false);
            m.blend = Engine::BlendMode::Additive;
            m.discardBelowCutoff = true;
            return m;
        };
        for (const auto& b : treeBatches_) {
            e.trees.push_back(card("tree", b.texture, false));
            e.litTrees.push_back(litCard(b.texture));
        }
        for (const auto& b : signBatches_) {
            e.signs.push_back(card("sign", b.texture, true));
            e.litSigns.push_back(litCard(b.texture));
        }

        // Street lamp light: glow quads added on top of the scene, without fog.
        e.lamp.name = "lamp";
        e.lamp.shading = Engine::ShadingModel::Unlit;
        e.lamp.blend = Engine::BlendMode::Additive;
        e.lamp.vertexColors = true;
        e.lamp.doubleSided = true;
        e.lamp.fog = false;
        e.lamp.baseTexture = glowTexture_.get();
        e.lamp.sampler = &SamplerState::LinearClamp;
    }

    void WorldRenderer::ApplyLighting()
    {
        bakedScale_ = rig_.BakedLightingScale(bakeRig_);
        lampFactor_ = rig_.LampFactor();
        SetLighting(engine_->renderer.Lighting(), rig_);
    }

    WorldRenderer::~WorldRenderer()
    {
        if (bakeJob_ && bakeJob_->worker.joinable()) bakeJob_->worker.join();
    }

    void WorldRenderer::UpdateSunShadows(GraphicsDevice& device, const Vector3& sunDirection,
                                         const float sunElevationDeg)
    {
#if defined(__EMSCRIPTEN__)
        // No worker threads in the browser build: the shadows stay as baked at load.
        (void)device;
        (void)sunDirection;
        (void)sunElevationDeg;
        return;
#else
        if (bakeJob_) {
            if (!bakeJob_->done.load()) return;
            if (bakeJob_->worker.joinable()) bakeJob_->worker.join();
            // The worker has prepared the mip chain and packed RGBA bytes. Submit all
            // levels together so the new texture is fully defined before it is bound.
            if (swapNext_ == 0 && !bakeJob_->macroLevels.empty()) {
                const auto& levels = bakeJob_->macroLevels;
                auto nextMacro = std::make_unique<Texture2D>(
                    device, levels[0].width, levels[0].height, true, SurfaceFormat::Color);
                for (std::size_t i = 0; i < levels.size(); ++i) {
                    const auto& level = levels[i];
                    nextMacro->SetData(static_cast<int>(i), nullptr,
                        level.rgba.data(), 0, static_cast<int>(level.rgba.size()));
                }
                macro_ = std::move(nextMacro);   // the terrain material picks it up at the next Draw
            }
            // Replace the road colours in small batches after the terrain texture.
            constexpr std::size_t kBatchesPerFrame = 24;
            const std::size_t end = std::min(roadBatches_.size(), swapNext_ + kBatchesPerFrame);
            for (; swapNext_ < end; ++swapNext_) {
                Batch& b = roadBatches_[swapNext_];
                if (!b.source || swapNext_ >= bakeJob_->colours.size() || bakeJob_->colours[swapNext_].empty()) continue;
                MeshData m = *b.source;
                const auto& colours = bakeJob_->colours[swapNext_];
                for (std::size_t i = 0; i < m.vertices.size() && i < colours.size(); ++i) m.vertices[i].color = colours[i];
                b.mesh = GpuMesh::Create(device, m, VertexLayout::PositionColorTexture);
            }
            if (swapNext_ >= roadBatches_.size()) {
                std::cout << "shadows: re-baked for the sun at " << std::lround(std::asin(std::clamp(-bakeJob_->sun.Y, -1.0f, 1.0f)) * 57.2958f)
                          << " degrees in " << bakeJob_->seconds << " s\n";
                shadowSun_ = bakeJob_->sun;
                bakeJob_.reset();
                swapNext_ = 0;
            }
            return;
        }
        // LightingRig switches its key direction to the moon at night. Use the actual solar
        // elevation so that moonlight does not trigger a costly new *sun* shadow bake.
        // A bake already running above can still finish and swap in after sunset.
        if (sunElevationDeg < 3.5f) return;
        const Vector3 toSun = -sunDirection;
        if (toSun.Y < 0.06f) return;
        const float turned = std::acos(std::clamp(Vector3::Dot(sunDirection, shadowSun_), -1.0f, 1.0f));
        if (turned < 10.0f * 3.14159265f / 180.0f) return;

        bakeJob_ = std::make_unique<ShadowBakeJob>();
        bakeJob_->sun = sunDirection;
        swapNext_ = 0;
        std::vector<std::pair<std::shared_ptr<const MeshData>, bool>> sources;
        sources.reserve(roadBatches_.size());
        for (const auto& b : roadBatches_) sources.emplace_back(b.source, b.verge);
        ShadowBakeJob* job = bakeJob_.get();
        job->worker = std::thread([this, job, sources = std::move(sources)]() {
            const auto started = std::chrono::steady_clock::now();
            const auto& terrain = world_.Terrain();
            const int macroW = std::min(2048, std::max(1, 2 * terrain.Columns()));
            const int macroH = std::min(2048, std::max(1, 2 * terrain.Rows()));
            const Image shadow = GroundShadows::Bake(world_, job->sun, macroW, macroH);
            Image macro(macroW, macroH);
            Image tint(macroW, macroH);
            ComputeMacro(shadow, macro, tint);
            job->colours.resize(sources.size());
            for (std::size_t i = 0; i < sources.size(); ++i) {
                if (!sources[i].first) continue;
                MeshData m = *sources[i].first;
                BakeRoadColours(m, shadow, sources[i].second ? &tint : nullptr);
                auto& out = job->colours[i];
                out.reserve(m.vertices.size());
                for (const auto& v : m.vertices) out.push_back(v.color);
            }
            // Downsample and pack on the worker. cna-engine's mip chain is already RGBA8
            // bytes, which the byte SetData overload copies on the update thread as they are.
            for (auto& level : CnaEngine::Graphics::BuildMipChain(macro.ToEngine())) {
                ShadowBakeJob::MacroLevel packed;
                packed.width = level.Width();
                packed.height = level.Height();
                packed.rgba = std::move(level.Bytes());
                job->macroLevels.push_back(std::move(packed));
            }
            job->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            job->done.store(true);
        });
#endif
    }

    void WorldRenderer::SetHeadlights(const Vector3& position, const Vector3& forward,
                                      const float intensity, const bool highBeam)
    {
        headlightPosition_ = position;
        headlightForward_ = forward;
        headlightForward_.Y = 0.0f;
        if (headlightForward_.LengthSquared() > 1e-6f) headlightForward_.Normalize();
        headlightIntensity_ = std::clamp(intensity, 0.0f, 1.0f);
        headlightHighBeam_ = highBeam;
    }

    void WorldRenderer::DrawLampLights()
    {
        if (lampFactor_ <= 0.01f || lampLights_.empty()) {
            return;
        }
        EnginePasses& e = *engine_;
        // Sodium-white lamps. The lantern runs near white; the pool on the ground carries its
        // lower intensity in the vertex colour, since both share this one material.
        e.lamp.diffuseColor = Vector3(0.95f, 0.84f, 0.60f) * lampFactor_;
        for (const auto& mesh : lampLights_) {
            if (mesh && Visible(e.camera, mesh->Sphere())) {
                Submit(e.queue, *mesh, e.lamp, mesh->Sphere());
            }
        }
        Render(e.renderer, e.queue, e.camera, stats_);
    }

    void WorldRenderer::DrawHeadlightFill()
    {
        if (headlightIntensity_ <= 0.01f) return;
        EnginePasses& e = *engine_;
        const float range = headlightHighBeam_ ? 275.0f : 165.0f;
        const auto inBeam = [&](const BoundingSphere& sphere) {
            const Vector3 delta = sphere.Center - headlightPosition_;
            const float along = Vector3::Dot(delta, headlightForward_);
            if (along + sphere.Radius < 0.0f || along - sphere.Radius > range) return false;
            const float sideways = std::sqrt(std::max(0.0f, delta.X * delta.X + delta.Z * delta.Z - along * along));
            return sideways < sphere.Radius + 5.0f + std::max(0.0f, along) * (headlightHighBeam_ ? 0.15f : 0.22f);
        };

        // Vertical scenery inside the beam, lit by the headlamps alone and added on top.
        auto& object = e.beamObjects.Lighting();
        object.defaultLighting = false;
        object.lightDirection = headlightForward_;
        object.lightColor = (headlightHighBeam_ ? Vector3(0.72f, 0.70f, 0.63f) : Vector3(0.61f, 0.57f, 0.49f)) *
                            headlightIntensity_;
        object.fillLights.clear();
        object.ambientColor = Vector3(0.0f, 0.0f, 0.0f);
        object.fogEnabled = true;
        object.fogColor = Vector3(0.0f, 0.0f, 0.0f);   // additive: fog fades it out
        object.fogStart = 10.0f;
        object.fogEnd = range;
        for (std::size_t i = 0; i < objectBatches_.size(); ++i) {
            const auto& batch = objectBatches_[i];
            if (!batch.mesh || !Visible(e.camera, batch.mesh->Sphere()) || !inBeam(batch.mesh->Sphere())) continue;
            if (batch.treeTrunk && rig_.fogEnd < 400.0f &&
                Vector3::Distance(batch.mesh->Sphere().Center, headlightPosition_) > rig_.fogEnd + 30.0f) continue;
            Submit(e.beamQueue, *batch.mesh, e.litObjects[i], batch.mesh->Sphere());
        }
        Render(e.beamObjects, e.beamQueue, e.camera, stats_);

        auto& cards = e.beamCards.Lighting();
        cards.fogEnabled = true;
        cards.fogColor = Vector3(0.0f, 0.0f, 0.0f);
        cards.fogStart = 6.0f;
        cards.fogEnd = range;
        const Vector3 treeLight = (headlightHighBeam_ ? Vector3(0.40f, 0.39f, 0.34f) : Vector3(0.32f, 0.30f, 0.26f)) *
                                  headlightIntensity_;
        const Vector3 signLight = (headlightHighBeam_ ? Vector3(0.59f, 0.56f, 0.48f) : Vector3(0.50f, 0.46f, 0.39f)) *
                                  headlightIntensity_;
        for (std::size_t i = 0; i < treeBatches_.size(); ++i) {
            const auto& batch = treeBatches_[i];
            if (!batch.mesh || !Visible(e.camera, batch.mesh->Sphere()) || !inBeam(batch.mesh->Sphere())) continue;
            e.litTrees[i].baseTexture = batch.texture;
            e.litTrees[i].diffuseColor = treeLight;
            Submit(e.beamQueue, *batch.mesh, e.litTrees[i], batch.mesh->Sphere());
        }
        for (std::size_t i = 0; i < signBatches_.size(); ++i) {
            const auto& batch = signBatches_[i];
            if (!batch.mesh || !Visible(e.camera, batch.mesh->Sphere()) || !inBeam(batch.mesh->Sphere())) continue;
            e.litSigns[i].baseTexture = batch.texture;
            e.litSigns[i].diffuseColor = signLight;
            Submit(e.beamQueue, *batch.mesh, e.litSigns[i], batch.mesh->Sphere());
        }
        Render(e.beamCards, e.beamQueue, e.camera, stats_);
    }

    Texture2D* WorldRenderer::TextureFor(const Surface s) const
    {
        switch (s) {
            case Surface::Asphalt: return asphalt_.get();
            case Surface::Gravel: return gravel_.get();
            case Surface::Paving: return paving_.get();
            case Surface::Concrete: return concrete_.get();
            case Surface::Marking: return marking_.get();
            case Surface::Grass: return grass_.get();
            case Surface::Cobbles: return cobbles_.get();
        }
        return white_.get();
    }

    void WorldRenderer::Draw(GraphicsDevice& device, const Matrix& view, const Matrix& projection, const BoundingFrustum& frustum,
                             const bool mirrored, const float maxDistance)
    {
        EnginePasses& e = *engine_;
        // The engine draws with exactly these matrices. A mirror pass arrives with its image
        // flipped already; the engine flips it itself (and culls the other winding), so it is
        // given the unflipped projection, whose frustum is `frustum`.
        e.camera.SetMatrices(view, mirrored ? projection * Matrix::CreateScale(-1.0f, 1.0f, 1.0f) : projection);
        e.camera.SetMirrored(mirrored);
        const RasterizerState& solid = mirrored ? RasterizerState::CullClockwise : RasterizerState::CullCounterClockwise;
        // Everything this pass may draw stops here.
        // The quality tier scales the main view's distances; a pass with its own cap (the mirror)
        // takes the shorter of the two.
        const float scale = maxDistance > 0.0f ? 1.0f : drawDistanceScale_;
        const float horizon = maxDistance > 0.0f ? std::min(maxDistance, rig_.fogEnd) : rig_.fogEnd * scale;
        const float terrainCull = maxDistance > 0.0f ? std::min(maxDistance, terrainCullDistanceM) : terrainCullDistanceM * scale;
        stats_.terrainChunksDrawn = 0;
        stats_.roadBatchesDrawn = 0;
        stats_.objectBatchesDrawn = 0;
        stats_.treeBatchesDrawn = 0;
        stats_.drawCalls = 0;
        stats_.triangles = 0;

        // The state the weather layers start from; the engine sets its own for each segment
        // and restores this one afterwards.
        device.setBlendStateProperty(BlendState::Opaque);
        device.setDepthStencilStateProperty(DepthStencilState::Default);
        device.setRasterizerStateProperty(solid);
        device.getSamplerStatesProperty()[0] = SamplerState::AnisotropicWrap;
        device.getSamplerStatesProperty()[1] = SamplerState::LinearClamp;

        // Wet ground, grass and walls read darker: water fills the pores and the fine texture
        // that scattered light back. Applied per frame on top of the rig's scale, since the
        // wetness moves continuously while the lighting refreshes in steps.
        const float wetGround = 1.0f - 0.16f * wetness_;
        const Vector3 eye = e.camera.Position();
        // Lying snow: the same surfaces again, alpha-blended white, pulled forward by the
        // marking depth bias so they win the depth test against themselves.
        const auto beginSnow = [&](const float alpha) {
            const Vector3 tint(std::min(1.0f, bakedScale_.X * 1.55f), std::min(1.0f, bakedScale_.Y * 1.58f),
                               std::min(1.0f, bakedScale_.Z * 1.66f));
            const float opacity = std::clamp(alpha, 0.0f, 1.0f);
            snowEffect_->setWorldProperty(Matrix::getIdentityProperty());
            snowEffect_->setViewProperty(view);
            snowEffect_->setProjectionProperty(projection);
            snowEffect_->setTextureProperty(snowTexture_.get());
            // Snow reflects most of the light the baked ground absorbed: well above the grass.
            snowEffect_->setDiffuseColorProperty(tint);
            snowEffect_->setAlphaProperty(opacity);
            snowEffect_->setFogColorProperty(rig_.fogColor);
            snowEffect_->setFogStartProperty(rig_.fogStart);
            snowEffect_->setFogEndProperty(rig_.fogEnd);
            device.setBlendStateProperty(BlendState::NonPremultiplied);
            device.setDepthStencilStateProperty(DepthStencilState::DepthRead);
            device.setRasterizerStateProperty(mirrored ? *markingStateMirrored_ : *markingState_);
            device.getSamplerStatesProperty()[0] = SamplerState::LinearWrap;
        };
        const auto endSnow = [&]() {
            device.setBlendStateProperty(BlendState::Opaque);
            device.setDepthStencilStateProperty(DepthStencilState::Default);
            device.setRasterizerStateProperty(solid);
            device.getSamplerStatesProperty()[0] = SamplerState::AnisotropicWrap;
        };
        const bool snowing = snow_ > 0.01f && snowEffect_ && snowTexture_;
        // At full cover the terrain already supplies snow beneath the grass verge.
        // Let it meet the gravel shoulder directly instead of layering two snow patterns.
        const bool fullSnowGround = snowing && snow_ >= 0.98f;

        // Terrain: the skirt, and the chunks within reach at their distance's level of detail.
        e.terrain.diffuseColor = bakedScale_ * wetGround;
        e.terrain.secondaryTexture = macro_.get();
        std::vector<const GpuMesh*> snowTerrain;
        if (terrainSkirt_) {
            Submit(e.queue, *terrainSkirt_, e.terrain, terrainSkirt_->Sphere());
        }
        for (const auto& chunk : terrainChunks_) {
            if (!chunk.lod0) {
                continue;
            }
            const float distance = Vector3::Distance(eye, chunk.centre) - chunk.radius;
            if (distance > terrainCull || !Visible(e.camera, chunk.lod0->Sphere())) {
                continue;
            }
            const int level = e.terrainLod.Select(distance);
            const GpuMesh* mesh = level == 0 ? chunk.lod0.get() : (level == 1 ? chunk.lod1.get() : chunk.lod2.get());
            if (!mesh) mesh = chunk.lod0.get();
            Submit(e.queue, *mesh, e.terrain, chunk.lod0->Sphere());
            ++stats_.terrainChunksDrawn;
            if (snowing) {
                const GpuMesh* snowMesh = level == 0 ? chunk.snowLod0.get() : (level == 1 ? chunk.snowLod1.get() : chunk.snowLod2.get());
                snowTerrain.push_back(snowMesh ? snowMesh : chunk.snowLod0.get());
            }
        }
        Render(e.renderer, e.queue, e.camera, stats_);
        if (snowing) {
            beginSnow(snow_);
            for (const GpuMesh* mesh : snowTerrain) {
                ApplyAll(*snowEffect_, device, *mesh);
                ++stats_.drawCalls;
            }
            endSnow();
        }

        // Roads and verges, then the markings.
        const Vector3 sky = rig_.horizonColor;
        for (int i = 0; i < kSurfaceCount; ++i) {
            const auto surface = static_cast<Surface>(i);
            Vector3 t(1.0f, 1.0f, 1.0f);
            if (surface == Surface::Marking) {
                t = Vector3(0.92f, 0.92f, 0.90f);
                if (wetness_ > 0.0f) {
                    // Wet asphalt swallows light and takes on the colour of the sky it reflects;
                    // the paint on it darkens less, because it stays rough.
                    const float wet = wetness_ * 0.45f;
                    t = Vector3(t.X * (1.0f - 0.42f * wet) + sky.X * 0.10f * wet, t.Y * (1.0f - 0.42f * wet) + sky.Y * 0.10f * wet,
                                t.Z * (1.0f - 0.40f * wet) + sky.Z * 0.13f * wet);
                }
            } else if (wetness_ > 0.0f) {
                // Wet asphalt and stone swallow light and take on the colour of the sky they
                // reflect; grass only darkens, like the terrain beside it.
                if (surface == Surface::Grass) {
                    t = Vector3(wetGround, wetGround, wetGround);
                } else {
                    const float wet = wetness_ * (surface == Surface::Asphalt ? 1.0f : 0.8f);
                    t = Vector3(1.0f - 0.42f * wet + sky.X * 0.10f * wet, 1.0f - 0.42f * wet + sky.Y * 0.10f * wet,
                                1.0f - 0.40f * wet + sky.Z * 0.13f * wet);
                }
            }
            e.surfaces[static_cast<std::size_t>(i)].diffuseColor =
                Vector3(t.X * bakedScale_.X, t.Y * bakedScale_.Y, t.Z * bakedScale_.Z);
        }
        for (const auto& b : roadBatches_) {
            if (!b.mesh || (fullSnowGround && b.verge) || !Visible(e.camera, b.mesh->Sphere())) {
                continue;
            }
            Submit(e.queue, *b.mesh, e.surfaces[static_cast<std::size_t>(b.surface)], b.mesh->Sphere(),
                   b.surface == Surface::Marking ? 1 : 0);
            ++stats_.roadBatchesDrawn;
        }
        Render(e.renderer, e.queue, e.camera, stats_);
        // Wet sheen over the road surfaces (not the markings: paint stays rough when it is wet).
        if (wetness_ > 0.01f && roadSheenEffect_) {
            const float strength = 0.95f * wetness_;
            roadSheenEffect_->setWorldProperty(Matrix::getIdentityProperty());
            roadSheenEffect_->setViewProperty(view);
            roadSheenEffect_->setProjectionProperty(projection);
            roadSheenEffect_->setFogColorProperty(Vector3(sky.X * strength, sky.Y * strength, sky.Z * strength));
            // The ramp: nothing under the bumper, full sheen by the time the road is a hundred
            // metres off, and never beyond the fog, where there is no road left to see.
            roadSheenEffect_->setFogStartProperty(7.0f);
            roadSheenEffect_->setFogEndProperty(std::min(150.0f, rig_.fogEnd));
            device.setRasterizerStateProperty(solid);
            device.setBlendStateProperty(BlendState::Additive);
            device.setDepthStencilStateProperty(DepthStencilState::DepthRead);
            for (const auto& b : roadBatches_) {
                if (b.surface == Surface::Marking || b.surface == Surface::Gravel || b.surface == Surface::Grass || !b.mesh ||
                    !frustum.Intersects(b.mesh->Sphere())) {
                    continue;
                }
                // Smooth asphalt mirrors the most; slabs, kerbs and cobbles are rougher and
                // drain between their joints.
                const float surfaceShare = b.surface == Surface::Asphalt ? 1.0f : (b.surface == Surface::Cobbles ? 0.55f : 0.75f);
                roadSheenEffect_->setFogColorProperty(sky * (strength * surfaceShare));
                ApplyAll(*roadSheenEffect_, device, *b.mesh);
                ++stats_.drawCalls;
                stats_.triangles += b.mesh->PrimitiveCount();
            }
            // Standing water once the rain has soaked the road: brighter patches of reflected
            // sky, fading out with distance like the sheen.
            const float standing = std::clamp((wetness_ - 0.35f) / 0.65f, 0.0f, 1.0f);
            if (standing > 0.0f && puddleEffect_ && puddleMask_) {
                puddleEffect_->setWorldProperty(Matrix::getIdentityProperty());
                puddleEffect_->setViewProperty(view);
                puddleEffect_->setProjectionProperty(projection);
                puddleEffect_->setTextureProperty(puddleMask_.get());
                puddleEffect_->setDiffuseColorProperty(Vector3(0.015f, 0.015f, 0.02f) + sky * (0.20f * standing));
                puddleEffect_->setFogStartProperty(6.0f);
                puddleEffect_->setFogEndProperty(std::min(120.0f, rig_.fogEnd));
                // The puddles lie on the road surface itself; the marking state's depth bias
                // keeps them from fighting it for the depth test.
                device.setRasterizerStateProperty(mirrored ? *markingStateMirrored_ : *markingState_);
                for (const auto& b : roadBatches_) {
                    if (!b.puddles || !frustum.Intersects(b.puddles->Sphere())) {
                        continue;
                    }
                    ApplyAll(*puddleEffect_, device, *b.puddles);
                    ++stats_.drawCalls;
                    stats_.triangles += b.puddles->PrimitiveCount();
                }
                device.setRasterizerStateProperty(solid);
            }
            device.setBlendStateProperty(BlendState::Opaque);
            device.setDepthStencilStateProperty(DepthStencilState::Default);
        }

        // Snow on the roads and pavements: a little thinner than on the fields, where the
        // traffic has worn it into slush.
        if (snowing) {
            beginSnow(snow_ * 0.78f);
            for (const auto& b : roadBatches_) {
                if (!b.mesh || !frustum.Intersects(b.mesh->Sphere())) continue;
                if (fullSnowGround && b.verge) continue;
                // Traffic wears snow thin on pavement. The grass verge belongs to the
                // adjoining field, while loose gravel retains more cover than asphalt.
                // Match those surfaces to their surroundings without another pass.
                const float cover = b.surface == Surface::Grass ? snow_ :
                                    b.surface == Surface::Gravel ? snow_ * 0.92f : snow_ * 0.78f;
                snowEffect_->setAlphaProperty(std::clamp(cover, 0.0f, 1.0f));
                ApplyAll(*snowEffect_, device, b.snowBase ? *b.snowBase : *b.mesh);
                ++stats_.drawCalls;
            }
            endSnow();
        }

        // Static objects: buildings, props, trunks (lit, textured).
        for (std::size_t i = 0; i < objectBatches_.size(); ++i) {
            const auto& b = objectBatches_[i];
            if (!b.mesh) {
                continue;
            }
            // Nothing past the fog end can be told from the fog itself, so no batch is drawn
            // beyond it; detail batches keep their own shorter range. Without this the buildings
            // of a settlement three kilometres away are still submitted in full.
            const float cull = b.cullDistance > 0.0f ? std::min(b.cullDistance, horizon) : horizon;
            if (!Visible(e.camera, b.mesh->Sphere(), cull)) {
                continue;
            }
            if (b.treeTrunk && rig_.fogEnd < 400.0f &&
                Vector3::Distance(eye, b.mesh->Sphere().Center) > horizon + 30.0f) continue;
            // Rain-soaked plaster, roof tiles and wood darken and pick up a wet gloss.
            Engine::Material& m = e.objects[i];
            m.diffuseColor = b.diffuse * (1.0f - 0.22f * wetness_);
            m.emissiveColor = b.emissive + b.nightEmissive * lampFactor_;
            m.specularColor = b.specular + Vector3(0.20f, 0.20f, 0.21f) * wetness_;
            m.specularPower = b.specularPower + 20.0f * wetness_;
            Submit(e.queue, *b.mesh, m, b.mesh->Sphere());
            ++stats_.objectBatchesDrawn;
        }
        Render(e.renderer, e.queue, e.camera, stats_);
        // Snow on the roofs.
        if (snowing) {
            beginSnow(snow_);
            for (const auto& b : objectBatches_) {
                if (!b.roof || !b.mesh || !frustum.Intersects(b.mesh->Sphere())) continue;
                if (Vector3::Distance(eye, b.mesh->Sphere().Center) - b.mesh->Sphere().Radius > horizon) continue;
                ApplyAll(*snowEffect_, device, *b.mesh);
                ++stats_.drawCalls;
            }
            endSnow();
        }

        // Trees: alpha-tested cards, both windings present, distance culled.
        // Alpha-tested foliage otherwise leaves pale cut-out silhouettes after the rest of the
        // world has vanished into dense fog. Use the same visibility horizon as opaque objects;
        // the fog ramp hides the last visible cards before a batch is culled.
        const float treeRange = std::min(horizon, maxDistance > 0.0f ? std::min(maxDistance, 1100.0f) : 1100.0f * scale * vegetationScale_);
        const Vector3 cardLight = bakedScale_ * (1.0f - 0.10f * wetness_);
        for (std::size_t i = 0; i < treeBatches_.size(); ++i) {
            const auto& b = treeBatches_[i];
            if (!b.mesh || !Visible(e.camera, b.mesh->Sphere(), treeRange)) {
                continue;
            }
            if (rig_.fogEnd < 400.0f && Vector3::Distance(b.mesh->Sphere().Center, eye) > treeRange + 30.0f) {
                continue;   // a coarse 256 m chunk must not leave a white silhouette beyond dense fog
            }
            e.trees[i].baseTexture = b.texture;   // the seasonal atlases are recoloured in place
            e.trees[i].diffuseColor = cardLight;
            Submit(e.queue, *b.mesh, e.trees[i], b.mesh->Sphere());
            ++stats_.treeBatchesDrawn;
        }
        // Sign faces: the tree material, both sides drawn, no distance cull beyond the frustum.
        for (std::size_t i = 0; i < signBatches_.size(); ++i) {
            const auto& b = signBatches_[i];
            if (!b.mesh || !Visible(e.camera, b.mesh->Sphere())) {
                continue;
            }
            e.signs[i].baseTexture = b.texture;
            e.signs[i].diffuseColor = cardLight;
            Submit(e.queue, *b.mesh, e.signs[i], b.mesh->Sphere());
        }
        Render(e.renderer, e.queue, e.camera, stats_);

        if (!mirrored) DrawHeadlightFill();
        // Street lamp light last, so it adds on top of everything the pass has drawn.
        DrawLampLights();
        device.getSamplerStatesProperty()[0] = SamplerState::AnisotropicWrap;
        device.setRasterizerStateProperty(RasterizerState::CullCounterClockwise);
    }
}
