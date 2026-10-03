import os
import sys
import time
import math
import glob
import asyncio
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOM_PASSWORD = os.getenv("ROOM_PASSWORD", "arena123")

MAP_WIDTH = 50.0
MAP_HEIGHT = 300.0
TPS = 30
PLAYER_SPEED = 25.0 / TPS
BULLET_SPEED = 50.0 / TPS
FIRE_COOLDOWN = int(0.5 * TPS)
HITBOX_RADIUS = 1.6

# --- Self-Contained Neural Model ---
try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical

    class ActorCritic(nn.Module):
        def __init__(self, state_dim=44):
            super().__init__()
            self.actor_backbone = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU()
            )
            self.move_head = nn.Linear(128, 5)

        def forward(self, state):
            return torch.clamp(self.move_head(self.actor_backbone(state)), -10.0, 10.0)

        def act(self, state_tensor):
            with torch.no_grad():
                logits = self.forward(state_tensor)
                return Categorical(logits=logits).sample().item()
except Exception:
    ActorCritic = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Client:
    def __init__(self, ws: WebSocket, username: str):
        self.ws = ws
        self.username = username
        self.keys = {"left": False, "right": False, "up": False, "down": False}

class GameServer:
    def __init__(self):
        self.p1: Optional[Client] = None
        self.p2: Optional[Client] = None
        self.running = False
        self.bullets = []
        self.loop_task: Optional[asyncio.Task] = None

        # Coordinates
        self.pA_x = 25.0
        self.pA_y = 20.0
        self.pA_score = 0
        self.pA_cooldown = 0

        self.pB_x = 25.0
        self.pB_y = 280.0
        self.pB_score = 0
        self.pB_cooldown = 0

        # AI Model
        self.ai_model = None
        self.ai_name = "Reflex Bot"
        self.init_ai()

    def init_ai(self):
        if ActorCritic is None:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models = sorted(glob.glob(os.path.join(base_dir, "saved_models", "*.pt")), key=os.path.getctime, reverse=True)
        if models:
            try:
                m = ActorCritic(state_dim=44)
                m.load_state_dict(torch.load(models[0], map_location="cpu", weights_only=False), strict=False)
                m.eval()
                self.ai_model = m
                self.ai_name = os.path.basename(models[0])
                print(f"[AI] Loaded neural model: {self.ai_name}")
            except Exception as e:
                print(f"[AI] Model load error: {e}")

    def reset_positions_and_scores(self):
        self.pA_x, self.pA_y = 25.0, 20.0
        self.pB_x, self.pB_y = 25.0, 280.0
        self.pA_score = 0
        self.pB_score = 0
        self.pA_cooldown = 0
        self.pB_cooldown = 0
        self.bullets.clear()

    def get_state_payload(self):
        num_humans = (1 if self.p1 else 0) + (1 if self.p2 else 0)
        is_pvp = (num_humans == 2)

        name_a = self.p1.username if self.p1 else "Waiting..."
        name_b = self.p2.username if self.p2 else self.ai_name

        return {
            "type": "state",
            "is_pvp": is_pvp,
            "num_humans": num_humans,
            "running": self.running,
            "player_a": {"name": name_a, "x": self.pA_x, "y": self.pA_y, "score": self.pA_score},
            "player_b": {"name": name_b, "x": self.pB_x, "y": self.pB_y, "score": self.pB_score},
            "bullets": [{"x": b["x"], "y": b["y"], "owner": b["owner"]} for b in self.bullets]
        }

    async def broadcast(self):
        payload = self.get_state_payload()
        dead = []
        for client in [self.p1, self.p2]:
            if client:
                try:
                    await client.ws.send_json(payload)
                except Exception:
                    dead.append(client.ws)
        for ws in dead:
            await self.handle_disconnect(ws)

    async def handle_disconnect(self, ws: WebSocket):
        if self.p1 and self.p1.ws == ws:
            print(f"[Server] Player 1 ({self.p1.username}) left.")
            # Promote Player 2 to Player 1 if present
            self.p1 = self.p2
            self.p2 = None
        elif self.p2 and self.p2.ws == ws:
            print(f"[Server] Player 2 ({self.p2.username}) left.")
            self.p2 = None

        self.running = False
        self.reset_positions_and_scores()
        await self.broadcast()

    def get_ai_features(self):
        sx, sy = self.pB_x, MAP_HEIGHT - 1.0 - self.pB_y
        rx, ry = self.pA_x, MAP_HEIGHT - 1.0 - self.pA_y
        features = [sx / MAP_WIDTH, sy / 100.0, self.pB_cooldown / float(FIRE_COOLDOWN),
                    (rx - sx) / MAP_WIDTH, (ry - sy) / MAP_HEIGHT]
        threats = []
        for b in self.bullets:
            if b["owner"] == "A":
                bx, by = b["x"], MAP_HEIGHT - 1.0 - b["y"]
                threats.append((math.hypot(bx - sx, by - sy), bx, by))
        threats.sort(key=lambda item: item[0])
        for i in range(5):
            if i < len(threats):
                _, bx, by = threats[i]
                features.extend([(bx - sx) / MAP_WIDTH, (by - sy) / 100.0, 0.0, -1.0, 0.5, 1.0])
            else:
                features.extend([0.0] * 6)
        features.extend([0.0] * 9)
        return features

    async def game_loop(self):
        while self.running:
            if not self.p1:
                break

            # 1. Move Player A (Top Human) - STRICTLY CLAMPED TO TOP HALF [0, 99]
            keys_a = self.p1.keys
            dx_a, dy_a = 0.0, 0.0
            if keys_a["left"]: dx_a -= 1.0
            if keys_a["right"]: dx_a += 1.0
            if keys_a["up"]: dy_a -= 1.0
            if keys_a["down"]: dy_a += 1.0
            self.pA_x = max(0.0, min(MAP_WIDTH - 1.0, self.pA_x + dx_a * PLAYER_SPEED))
            self.pA_y = max(0.0, min(99.0, self.pA_y + dy_a * PLAYER_SPEED))

            # 2. Move Player B - STRICTLY CLAMPED TO BOTTOM HALF [200, 299]
            dx_b, dy_b = 0.0, 0.0
            if self.p2:
                # 2-Player Human PVP
                keys_b = self.p2.keys
                if keys_b["left"]: dx_b -= 1.0
                if keys_b["right"]: dx_b += 1.0
                if keys_b["up"]: dy_b -= 1.0
                if keys_b["down"]: dy_b += 1.0
            else:
                # Solo AI
                if self.ai_model:
                    try:
                        tb = torch.tensor(self.get_ai_features(), dtype=torch.float32).unsqueeze(0)
                        move_b = self.ai_model.act(tb)
                        if move_b == 1: dx_b = -1.0
                        elif move_b == 2: dx_b = 1.0
                        elif move_b == 3: dy_b = -1.0
                        elif move_b == 4: dy_b = 1.0
                    except Exception:
                        dx_b = 1.0 if self.pB_x < 25.0 else -1.0
                else:
                    # Smart evasive patrol if no model
                    incoming = [b for b in self.bullets if b["owner"] == "A" and b["y"] > 150]
                    if incoming and abs(incoming[0]["x"] - self.pB_x) < 5.0:
                        dx_b = 1.0 if incoming[0]["x"] <= self.pB_x else -1.0
                    else:
                        dx_b = 1.0 if self.pB_x < 15.0 else (-1.0 if self.pB_x > 35.0 else 0.0)

            self.pB_x = max(0.0, min(MAP_WIDTH - 1.0, self.pB_x + dx_b * PLAYER_SPEED))
            self.pB_y = max(200.0, min(299.0, self.pB_y + dy_b * PLAYER_SPEED))

            # 3. Auto-fire on cooldown
            if self.pA_cooldown > 0:
                self.pA_cooldown -= 1
            else:
                self.bullets.append({"x": self.pA_x, "y": self.pA_y, "vy": BULLET_SPEED, "owner": "A"})
                self.pA_cooldown = FIRE_COOLDOWN

            if self.pB_cooldown > 0:
                self.pB_cooldown -= 1
            else:
                self.bullets.append({"x": self.pB_x, "y": self.pB_y, "vy": -BULLET_SPEED, "owner": "B"})
                self.pB_cooldown = FIRE_COOLDOWN

            # 4. Bullet physics & collisions
            surviving = []
            for b in self.bullets:
                b["y"] += b["vy"]
                if not (0 <= b["y"] < MAP_HEIGHT):
                    continue

                if b["owner"] == "B" and math.hypot(b["x"] - self.pA_x, b["y"] - self.pA_y) < HITBOX_RADIUS:
                    self.pB_score += 1
                    continue
                if b["owner"] == "A" and math.hypot(b["x"] - self.pB_x, b["y"] - self.pB_y) < HITBOX_RADIUS:
                    self.pA_score += 1
                    continue

                surviving.append(b)

            self.bullets = surviving
            await self.broadcast()
            await asyncio.sleep(1.0 / TPS)

server = GameServer()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        init_data = await ws.receive_json()
        username = str(init_data.get("username", "")).strip()[:14] or "Player"
        password = str(init_data.get("password", ""))

        if password != ROOM_PASSWORD:
            await ws.send_json({"type": "error", "message": "Incorrect room password."})
            await ws.close(code=4002)
            return

        # Slot Assignment
        if server.p1 is None:
            server.p1 = Client(ws, username)
            slot = "A"
        elif server.p2 is None:
            # If same name typed, make it distinct
            if server.p1.username == username:
                username = f"{username}_2"
            server.p2 = Client(ws, username)
            slot = "B"
            # NEW CHALLENGER TAKEOVER: Halt any running solo match and reset to 0-0
            server.running = False
            server.reset_positions_and_scores()
            print(f"[Server] Challenger {username} entered! Switched to 2-Player Mode.")
        else:
            await ws.send_json({"type": "error", "message": "Room is full (Maximum 2 players)."})
            await ws.close(code=4001)
            return

        await ws.send_json({"type": "init", "slot": slot, "username": username})
        await server.broadcast()

        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "ping":
                await ws.send_json({"type": "pong"})

            elif mtype == "start":
                server.running = True
                server.reset_positions_and_scores()
                if server.loop_task is None or server.loop_task.done():
                    server.loop_task = asyncio.create_task(server.game_loop())

            elif mtype == "stop":
                server.running = False
                server.reset_positions_and_scores()
                await server.broadcast()

            elif mtype == "keys":
                client = server.p1 if ws == (server.p1.ws if server.p1 else None) else server.p2
                if client:
                    client.keys = {
                        "left": bool(msg.get("left", False)),
                        "right": bool(msg.get("right", False)),
                        "up": bool(msg.get("up", False)),
                        "down": bool(msg.get("down", False))
                    }

    except WebSocketDisconnect:
        await server.handle_disconnect(ws)
    except Exception as e:
        print(f"[Server Error] {e}")
        await server.handle_disconnect(ws)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
