#pragma once
#include "CoreMinimal.h"
#include "Misc/Optional.h"
#include "TaxiTypes.generated.h"

USTRUCT(BlueprintType)
struct FTaxiData {
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) FVector Position = FVector::ZeroVector;
    UPROPERTY(BlueprintReadOnly) FString Orientation = TEXT("North");
    UPROPERTY(BlueprintReadOnly) float BrainValue = 0.0f;
    UPROPERTY(BlueprintReadOnly) FVector4 Sensors = FVector4(0,0,0,0);
    
    // Scenario data
    UPROPERTY(BlueprintReadOnly) TArray<FVector> Obstacles;
    UPROPERTY(BlueprintReadOnly) FVector PassengerPos = FVector::ZeroVector;
    UPROPERTY(BlueprintReadOnly) FVector TargetPos = FVector::ZeroVector;
    UPROPERTY(BlueprintReadOnly) bool bHasPassenger = false;
    UPROPERTY(BlueprintReadOnly) bool bSuccess = false;
    UPROPERTY(BlueprintReadOnly) bool bIsStarting = false;
};
