
import asyncio
import websockets
import json

async def test_client():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # 1. Wait for Init (Handshake)
            init_msg = await websocket.recv()
            data = json.loads(init_msg)
            print(f"[RECV] Type: {data.get('type')}")
            if data.get('type') == 'init':
                print(f"  Grid Size: {data.get('grid_size')}")
                print(f"  Obstacles: {len(data.get('obstacles'))}")
            
            # 2. Receive a few steps
            for _ in range(5):
                msg = await websocket.recv()
                step_data = json.loads(msg)
                if step_data.get('type') == 'step':
                    print(f"[RECV] Step: {step_data.get('step')} | Pos: {step_data.get('position')} | Val: {step_data.get('brain').get('value'):.2f}")
                elif step_data.get('type') == 'episode_end':
                    print("[RECV] Episode End")
                    break
            
            print("Test passed. Closing connection.")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_client())
