#pragma once
#include "CoreMinimal.h"
#include "Misc/Optional.h"
#include "Engine/GameInstance.h"
#include "IWebSocket.h"
#include "TaxiTypes.h"
#include "TaxiGameInstance.generated.h"
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTaxiStepUpdate, FTaxiData, Data);
UCLASS()
class TAXIAGENTSIM_API UTaxiGameInstance : public UGameInstance {
    GENERATED_BODY()
public:
    virtual void Init() override;
    virtual void Shutdown() override;
    TSharedPtr<IWebSocket> WebSocket;
    UPROPERTY(BlueprintAssignable, Category = "Taxi")
    FOnTaxiStepUpdate OnTaxiStepUpdate;

    TArray<FVector> CachedObstacles;
};