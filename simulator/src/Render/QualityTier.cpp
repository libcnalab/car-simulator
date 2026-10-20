#include "CarSim/Render/QualityTier.hpp"

#include "CnaEngine/Graphics/QualitySchema.hpp"

#include <algorithm>
#include <cctype>

namespace CarSim::Render
{
    namespace
    {
        // The tiers as a cna-engine quality schema: each setting is declared with its bounds,
        // so a preset value outside them is rejected when the schema is built instead of
        // silently reaching the renderer.
        const CnaEngine::Graphics::QualitySchema& TierSchema()
        {
            static const CnaEngine::Graphics::QualitySchema schema = [] {
                const QualitySettings defaults;
                CnaEngine::Graphics::QualitySchema s;
                s.Declare({"drawDistanceScale", defaults.drawDistanceScale, 0.1, 4.0, {}});
                s.Declare({"mirrorDistanceM", defaults.mirrorDistanceM, 0.0, 5000.0, {}});
                s.Declare({"mirrorUpdateEvery", defaults.mirrorUpdateEvery, 1.0, 16.0, {}});
                s.Declare({"vegetationScale", defaults.vegetationScale, 0.0, 4.0, {}});
                // Half the world distance, a quarter of the vegetation, and a mirror that is
                // redrawn every third frame at a third of the distance. On a software rasteriser
                // this roughly halves the frame; on a GPU it is the setting for 1440p or higher.
                s.DefinePreset("low", {{"drawDistanceScale", 0.5f},
                                       {"mirrorDistanceM", 110.0f},
                                       {"mirrorUpdateEvery", 3},
                                       {"vegetationScale", 0.5f}});
                s.DefinePreset("medium", {{"drawDistanceScale", 0.75f},
                                          {"mirrorDistanceM", 200.0f},
                                          {"mirrorUpdateEvery", 2},
                                          {"vegetationScale", 0.75f}});
                // The defaults are the high tier.
                s.DefinePreset("high", {});
                return s;
            }();
            return schema;
        }
    }

    QualitySettings SettingsFor(const QualityTier tier)
    {
        // No setting has a device limit, so the profile does not change any value.
        const auto resolved = TierSchema().Resolve({ToString(tier), {}}, CnaEngine::Graphics::DeviceCapabilities{});
        return QualitySettings{resolved.Float("drawDistanceScale"), resolved.Float("mirrorDistanceM"),
                               resolved.Int("mirrorUpdateEvery"), resolved.Float("vegetationScale")};
    }

    const char* ToString(const QualityTier tier)
    {
        switch (tier) {
            case QualityTier::Low: return "low";
            case QualityTier::Medium: return "medium";
            case QualityTier::High: return "high";
        }
        return "high";
    }

    bool QualityFromName(const std::string& name, QualityTier& out)
    {
        std::string key;
        key.reserve(name.size());
        for (const char c : name) {
            key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (key == "low") { out = QualityTier::Low; return true; }
        if (key == "medium" || key == "mid") { out = QualityTier::Medium; return true; }
        if (key == "high" || key == "full") { out = QualityTier::High; return true; }
        return false;
    }
}
