#include "TaxiAgentActor.h"
#include "TaxiGameInstance.h"
#include "Components/StaticMeshComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "Camera/CameraComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "Misc/Optional.h"

ATaxiAgentActor::ATaxiAgentActor() {
    PrimaryActorTick.bCanEverTick = true;

    Visual = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Visual"));
    RootComponent = Visual;

    SpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("SpringArm"));
    SpringArm->SetupAttachment(RootComponent);
    SpringArm->TargetArmLength = 1500.0f;
    SpringArm->SetRelativeRotation(FRotator(-35.0f, 0.0f, 0.0f));
    SpringArm->bDoCollisionTest = false;

    Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
    Camera->SetupAttachment(SpringArm);

    AutoPossessPlayer = EAutoReceiveInput::Player0;

    static ConstructorHelpers::FObjectFinder<UStaticMesh> CarMesh(TEXT("/Game/Vehicles/SportsCar/SM_SportsCar.SM_SportsCar"));
    if (CarMesh.Succeeded()) Visual->SetStaticMesh(CarMesh.Object);
    
    static ConstructorHelpers::FObjectFinder<UStaticMesh> OffMesh(TEXT("/Game/Vehicles/OffroadCar/SM_Offroad_Body.SM_Offroad_Body"));
    if (OffMesh.Succeeded()) ObstacleMesh = OffMesh.Object;
    
    InterpSpeed = 5.0f; // Smoother "physics" feel
}

void ATaxiAgentActor::BeginPlay() {
    Super::BeginPlay();
    TargetLoc = GetActorLocation();
    TargetRot = GetActorRotation();
    if (UTaxiGameInstance* GI = Cast<UTaxiGameInstance>(GetGameInstance()))
        GI->OnTaxiStepUpdate.AddDynamic(this, &ATaxiAgentActor::OnDataUpdate);
}

void ATaxiAgentActor::Tick(float DT) {
    Super::Tick(DT);
    SetActorLocation(FMath::VInterpTo(GetActorLocation(), TargetLoc, DT, InterpSpeed));
    SetActorRotation(FMath::RInterpTo(GetActorRotation(), TargetRot, DT, InterpSpeed));
    
    if (CelebrationTimer > 0) CelebrationTimer -= DT;
    if (StartTimer > 0) StartTimer -= DT;

    DrawSensors();
    DrawScenario();
}

void ATaxiAgentActor::OnDataUpdate(FTaxiData Data) {
    CurrentData = Data;
    TargetLoc = Data.Position;
    
    if (Data.bSuccess) CelebrationTimer = 2.0f;
    if (Data.bIsStarting) StartTimer = 2.0f;
    
    float Yaw = 0;
    if (Data.Orientation == TEXT("South")) Yaw = 90;
    else if (Data.Orientation == TEXT("North")) Yaw = -90;
    else if (Data.Orientation == TEXT("East")) Yaw = 0;
    else if (Data.Orientation == TEXT("West")) Yaw = 180;
    
    TargetRot = FRotator(0, Yaw, 0);
}

void ATaxiAgentActor::DrawSensors() {
    FVector Start = GetActorLocation() + FVector(0,0,100);
    float Range = 750.0f;
    if (CurrentData.Sensors.X > 0) DrawDebugLine(GetWorld(), Start, Start + FVector(-Range, 0, 0), FColor::Red, false, -1, 0, 15.f);
    if (CurrentData.Sensors.Y > 0) DrawDebugLine(GetWorld(), Start, Start + FVector(Range, 0, 0), FColor::Red, false, -1, 0, 15.f);
    if (CurrentData.Sensors.Z > 0) DrawDebugLine(GetWorld(), Start, Start + FVector(0, Range, 0), FColor::Red, false, -1, 0, 15.f);
    if (CurrentData.Sensors.W > 0) DrawDebugLine(GetWorld(), Start, Start + FVector(0, -Range, 0), FColor::Red, false, -1, 0, 15.f);
}

void ATaxiAgentActor::DrawScenario() {
    const float CellSize = 500.0f;
    const float HalfCell = CellSize / 2.0f;

    // 1. Grid
    for(int i=0; i<=5; i++){
        DrawDebugLine(GetWorld(), FVector(i*CellSize-HalfCell, -HalfCell, 2), FVector(i*CellSize-HalfCell, 4.5f*CellSize, 2), FColor::Cyan, false, -1, 0, 3.f);
        DrawDebugLine(GetWorld(), FVector(-HalfCell, i*CellSize-HalfCell, 2), FVector(4.5f*CellSize, i*CellSize-HalfCell, 2), FColor::Cyan, false, -1, 0, 3.f);
    }

    // 2. 3D Obstacle Cars
    for(const FVector& Obs : CurrentData.Obstacles) {
        // Draw highly visible silver boxes that look like cars (lenghtened)
        // Use persistent=false, lifetime=-1, thickness=10
        DrawDebugBox(GetWorld(), Obs + FVector(0,0,100), FVector(240, 120, 100), FColor::Silver, false, -1, 0, 10.f);
        
        // Sphere inside to find it
        DrawDebugSphere(GetWorld(), Obs + FVector(0,0,110), 60.f, 8, FColor::Red, false, -1, 0, 4.f);
    }

    // 3. 3D Collectibles
    if(!CurrentData.bHasPassenger)
        DrawDebugSphere(GetWorld(), CurrentData.PassengerPos + FVector(0,0,100), 100.f, 16, FColor::Yellow, false, -1, 0, 10.f);
    
    DrawDebugBox(GetWorld(), CurrentData.TargetPos + FVector(0,0,20), FVector(245, 245, 20), FColor::Green, false, -1, 0, 12.f);

    // 4. Celebration Effect
    if (CelebrationTimer > 0) {
        FVector ActorLoc = GetActorLocation();
        
        // Rapidly flashing green box around taxi
        FColor FlashColor = (FMath::Sin(GetWorld()->GetTimeSeconds() * 20.0f) > 0) ? FColor::Green : FColor::White;
        DrawDebugBox(GetWorld(), ActorLoc + FVector(0,0,150), FVector(260, 140, 120), FlashColor, false, -1, 0, 25.f);

        // Huge "SUCCESS" text above the car
        DrawDebugString(GetWorld(), ActorLoc + FVector(0,0,500), TEXT("SUCCESS!"), nullptr, FColor::Green, 0.05f, true, 8.0f);

        // Confetti: Spawning random colored spheres around the car
        for (int i=0; i<20; i++) {
            FVector RandomDir = FMath::VRand() * FMath::RandRange(100.0f, 800.0f);
            RandomDir.Z = FMath::Abs(RandomDir.Z) + 50.0f; 
            DrawDebugSphere(GetWorld(), ActorLoc + RandomDir, 40.f, 8, FColor::MakeRandomColor(), false, 0.1f, 0, 6.f);
        }
    }

    // 5. Start Message
    if (StartTimer > 0) {
        FVector ActorLoc = GetActorLocation();
        DrawDebugString(GetWorld(), ActorLoc + FVector(0,0,500), TEXT("START!"), nullptr, FColor::Cyan, 0.05f, true, 8.0f);
        DrawDebugBox(GetWorld(), ActorLoc + FVector(0,0,150), FVector(300, 150, 150), FColor::White, false, -1, 0, 30.f);
    }
}