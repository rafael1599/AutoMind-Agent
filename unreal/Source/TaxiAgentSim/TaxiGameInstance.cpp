#include "TaxiGameInstance.h"
#include "WebSocketsModule.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Misc/Optional.h"

void UTaxiGameInstance::Init() {
    Super::Init();
    if (!FModuleManager::Get().IsModuleLoaded("WebSockets")) FModuleManager::Get().LoadModule("WebSockets");
    WebSocket = FWebSocketsModule::Get().CreateWebSocket("ws://localhost:8765");
    WebSocket->OnMessage().AddLambda([this](const FString& MessageString) {
        TSharedPtr<FJsonObject> Json;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(MessageString);
        if (FJsonSerializer::Deserialize(Reader, Json)) {
            FString MsgType = Json->GetStringField(TEXT("type"));
            
            if (MsgType == TEXT("config") || MsgType == TEXT("init")) {
                // Initialize World State (Obstacles)
                this->CachedObstacles.Empty();
                const TArray<TSharedPtr<FJsonValue>> ObsArray = Json->GetArrayField(TEXT("obstacles"));
                for (auto& Val : ObsArray) {
                    TSharedPtr<FJsonObject> Obs = Val->AsObject();
                    this->CachedObstacles.Add(FVector(Obs->GetNumberField(TEXT("x")) * 500.f, Obs->GetNumberField(TEXT("y")) * 500.f, 0));
                }
                UE_LOG(LogTemp, Warning, TEXT("[TaxiBridge] Scenario Initialized: %d obstacles"), this->CachedObstacles.Num());
                
                FTaxiData InitData;
                InitData.bIsStarting = true;
                InitData.Obstacles = this->CachedObstacles;
                OnTaxiStepUpdate.Broadcast(InitData);
            }
            else if (MsgType == TEXT("step")) {
                FTaxiData Data;
                Data.Obstacles = this->CachedObstacles;
                
                TSharedPtr<FJsonObject> Pos = Json->GetObjectField(TEXT("position"));
                TSharedPtr<FJsonObject> Bm = Json->GetObjectField(TEXT("brain"));
                TSharedPtr<FJsonObject> Sens = Json->GetObjectField(TEXT("sensors"));
                TSharedPtr<FJsonObject> Status = Json->GetObjectField(TEXT("status"));

                Data.Position = FVector(Pos->GetNumberField(TEXT("x")) * 500.f, Pos->GetNumberField(TEXT("y")) * 500.f, 20.0f);
                Data.Orientation = Pos->GetStringField(TEXT("orientation"));
                Data.BrainValue = Bm->GetNumberField(TEXT("value"));
                Data.bHasPassenger = Status->GetBoolField(TEXT("has_passenger"));

                TSharedPtr<FJsonObject> PPos = Status->GetObjectField(TEXT("passenger_pos"));
                TSharedPtr<FJsonObject> DPos = Status->GetObjectField(TEXT("dest_pos"));
                Data.PassengerPos = FVector(PPos->GetNumberField(TEXT("x")) * 500.f, PPos->GetNumberField(TEXT("y")) * 500.f, 0);
                Data.TargetPos = FVector(DPos->GetNumberField(TEXT("x")) * 500.f, DPos->GetNumberField(TEXT("y")) * 500.f, 0);

                Data.Sensors = FVector4(
                    Sens->GetNumberField(TEXT("N")), 
                    Sens->GetNumberField(TEXT("S")), 
                    Sens->GetNumberField(TEXT("E")), 
                    Sens->GetNumberField(TEXT("W"))
                );
                Data.bSuccess = Status->GetBoolField(TEXT("is_success"));
                OnTaxiStepUpdate.Broadcast(Data);
            }
            else if (MsgType == TEXT("episode_end")) {
                if (Json->GetBoolField(TEXT("success"))) {
                    FTaxiData SuccessData;
                    SuccessData.bSuccess = true;
                    OnTaxiStepUpdate.Broadcast(SuccessData);
                    UE_LOG(LogTemp, Warning, TEXT("[TaxiBridge] Episode Success!"));
                }
            }
        }
    });
    WebSocket->Connect();
}
void UTaxiGameInstance::Shutdown() {
    if (WebSocket.IsValid() && WebSocket->IsConnected()) WebSocket->Close();
    Super::Shutdown();
}