using UnrealBuildTool;

public class TaxiAgentSim : ModuleRules
{
public TaxiAgentSim(ReadOnlyTargetRules Target) : base(Target)
{
PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

PublicDependencyModuleNames.AddRange(new string[] { 
"Core", 
"CoreUObject", 
"Engine", 
"InputCore", 
"WebSockets", 
"Json", 
"JsonUtilities",
"Slate",
"SlateCore" 
});
}
}
