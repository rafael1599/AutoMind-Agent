#pragma once
#include "CoreMinimal.h"
#include "Misc/Optional.h"
#include "GameFramework/Pawn.h"
#include "TaxiTypes.h"
#include "TaxiAgentActor.generated.h"

UCLASS()
class TAXIAGENTSIM_API ATaxiAgentActor : public APawn {
    GENERATED_BODY()
public:	
    ATaxiAgentActor();
    virtual void Tick(float DeltaTime) override;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Taxi")
    class UStaticMeshComponent* Visual;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Taxi")
    class USpringArmComponent* SpringArm;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Taxi")
    class UCameraComponent* Camera;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Taxi")
    float InterpSpeed = 5.0f;

    UPROPERTY()
    class UStaticMesh* ObstacleMesh;

protected:
    virtual void BeginPlay() override;

private:
    FVector TargetLoc;
    FRotator TargetRot;
    FTaxiData CurrentData;
    float CelebrationTimer = 0.0f;
    float StartTimer = 0.0f;

    UFUNCTION() void OnDataUpdate(FTaxiData Data);
    void DrawSensors();
    void DrawScenario();
};