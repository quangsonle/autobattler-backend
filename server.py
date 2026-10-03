import os
import sys
import time
import math
import glob
import asyncio
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROOM_PASSWORD = os.getenv("ROOM_PASSWORD", "arena123")
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 900

MAP_WIDTH = 50.0
MAP_HEIGHT = 300.0
TPS = 30
PLAYER_SPEED = 25.0 / TPS
BULLET_SPEED = 50.0 / TPS
FIRE_COOLDOWN = int(0.5 * TPS)
HITBOX_RADIUS = 1.6

# --- SELF-CONTAINED NEURAL NETWORK (NO MODEL.PY NEEDED!) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim=44):
        super().__init__()
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.move_head = nn.Linear(128, 5)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        feat = self.actor_backbone(state)
        move_logits = torch.clamp(self.move_head(feat), -10.0, 10.0)
        value = self.critic(state)
        return move_logits, value

    def act(self, state_tensor, deterministic=False):
        with torch.no_grad():
            move_logits, value = self.forward(state_tensor)
            if deterministic:
                move_act = torch.argmax(move_logits, dim=-1).item()
            else:
                move_act = Categorical(logits=move_logits).sample().item()
            return move_act, value.squeeze().item()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

failed_attempts: Dict[str, Dict] = {}

class Player:
    def __init__(self, pid: str, username: str, x: float, y: float, y_min: float, y_max: float):
        self.pid = pid
        self.username = username
        self.x = x
        self.y = y
        self.y_min = y_min
        self.y_max = y_max
        self.score = 0
        self.cooldown = 0
        self.keys = {"left": False, "right": False, "up": False, "down": False}

    def move(self, move_act: Optional[int] = None):
        dx, dy = 0.0, 0.0
        if move_act is not None:
            if move_act == 1: dx = -1.0
            elif move_act == 2: dx = 1.0
            elif move_act == 3: dy = 1.0 if self.pid == 'A' else -1.0
            elif move_act == 4: dy = -1.0 if self.pid == 'A' else 1.0
        else:
            if self.keys["left"]: dx -= 1.0
            if self.keys["right"]: dx += 1.0
            if self.keys["up"]: dy -= 1.0
            if self.keys["down"]: dy += 1.0

        self.x = max(0.0, min(MAP_WIDTH - 1.0, self.x + dx * PLAYER_SPEED))
        self.y = max(self.y_min, min(self.y_max, self.y + dy * PLAYER_SPEED))

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1

class Bullet:
    def __init__(self, x: float, y: float, vy: float, owner: str):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = vy
        self.owner = owner

    def update(self) -> bool:
        self.y += self.vy
        return 0 <= self.y < MAP_HEIGHT

class GameLobby:
    def __init__(self):
        # Look for saved_models in current repo root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, "saved_models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.active_connections: Dict[WebSocket, Player] = {}
        self.bullets = []
        self.running = False
        self.loop_task: Optional[asyncio.Task] = None
        self.solo_mode = True

        self.loaded_model = None
        self.current_model_name = "Waiting for Model..."
        self.ai_player = Player("B", "Model", 25.0, 280.0, 200.0, 299.0)

        self.refresh_available_models()

    def refresh_available_models(self):
        files = sorted(glob.glob(os.path.join(self.models_dir, "*.pt")), key=os.path.getctime, reverse=True)
        self.available_models = [os.path.basename(f) for f in files]
        print(f"[Lobby] Found {len(self.available_models)} models in {self.models_dir}: {self.available_models}")

        if self.available_models:
            self.load_ai_model(self.available_models[0])

    def load_ai_model(self, model_name: str) -> Tuple[bool, str]:
        path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(path):
            return False, f"File {model_name} not found."

        try:
            m = ActorCritic(state_dim=44)
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                state_dict = torch.load(path, map_location="cpu")

            m.load_state_dict(state_dict, strict=False)
            m.eval()
            self.loaded_model = m
            self.current_model_name = model_name
            self.ai_player.username = model_name
            print(f"[Lobby] Successfully loaded AI Model: {model_name}")
            return True, ""
        except Exception as e:
            print(f"[Lobby] Failed to load model {model_name}: {e}")
            return False, str(e)

    def get_canonical_features(self, pid: str) -> list:
        is_a = (pid == 'A')
        p_list = list(self.active_connections.values())
        self_p = p_list[0] if is_a else self.ai_player
        rival_p = self.ai_player if is_a else (p_list[0] if len(p_list) >= 1 else self.ai_player)

        def to_canonical(x, y, vy):
            return (x, y, vy) if is_a else (x, MAP_HEIGHT - 1.0 - y, -vy)

        sx, sy, _ = to_canonical(self_p.x, self_p.y, 0)
        rx, ry, _ = to_canonical(rival_p.x, rival_p.y, 0)

        features = [sx / MAP_WIDTH, sy / 100.0, self_p.cooldown / float(FIRE_COOLDOWN),
                    (rx - sx) / MAP_WIDTH, (ry - sy) / MAP_HEIGHT]

        enemy_bullets = []
        own_bullets = []
        for b in self.bullets:
            bx, by, bvy = to_canonical(b.x, b.y, b.vy)
            dist = math.hypot(bx - sx, by - sy)
            if b.owner != pid:
                enemy_bullets.append((dist, bx, by, bvy))
            else:
                own_bullets.append((dist, bx, by))

        enemy_bullets.sort(key=lambda item: item[0])
        for i in range(5):
            if i < len(enemy_bullets):
                _, bx, by, bvy = enemy_bullets[i]
                t_hit = (by - sy) / (-bvy) if bvy < -1e-4 else 10.0
                features.extend([(bx - sx) / MAP_WIDTH, (by - sy) / 100.0, 0.0, bvy / BULLET_SPEED,
                                 max(0.0, min(5.0, t_hit)) / 5.0, 1.0])
            else:
                features.extend([0.0] * 6)

        own_bullets.sort(key=lambda item: item[0])
        for i in range(3):
            if i < len(own_bullets):
                _, bx, by = own_bullets[i]
                features.extend([(bx - rx) / MAP_WIDTH, (by - ry) / MAP_HEIGHT, 1.0])
            else:
                features.extend([0.0] * 3)

        return features

    def reset_arena(self):
        self.bullets.clear()
        p_list = list(self.active_connections.values())
        if len(p_list) >= 1:
            p_list[0].x, p_list[0].y = 25.0, 20.0
            p_list[0].cooldown = 0
        if len(p_list) >= 2:
            p_list[1].x, p_list[1].y = 25.0, 280.0
            p_list[1].cooldown = 0
        self.ai_player.x, self.ai_player.y = 25.0, 280.0
        self.ai_player.cooldown = 0

    def get_current_state(self):
        p_list = list(self.active_connections.values())
        pA_name = p_list[0].username if len(p_list) >= 1 else "Waiting..."
        pA_x = p_list[0].x if len(p_list) >= 1 else 25.0
        pA_y = p_list[0].y if len(p_list) >= 1 else 20.0
        pA_score = p_list[0].score if len(p_list) >= 1 else 0

        if len(p_list) >= 2 and not self.solo_mode:
            pB_name = p_list[1].username
            pB_x = p_list[1].x
            pB_y = p_list[1].y
            pB_score = p_list[1].score
        else:
            pB_name = self.current_model_name
            pB_x = self.ai_player.x
            pB_y = self.ai_player.y
            pB_score = self.ai_player.score

        return {
            "type": "state",
            "player_a": {"x": pA_x, "y": pA_y, "score": pA_score, "name": pA_name},
            "player_b": {"x": pB_x, "y": pB_y, "score": pB_score, "name": pB_name},
            "bullets": [{"x": b.x, "y": b.y, "owner": b.owner} for b in self.bullets]
        }

    async def broadcast(self, data: dict):
        dead = []
        for ws in list(self.active_connections.keys()):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            p = self.active_connections.pop(ws)
            print(f"[Lobby] Session ended for {p.username}")
            self.running = False
            self.reset_arena()

    async def game_tick(self):
        while self.running:
            p_list = list(self.active_connections.values())
            if len(p_list) == 0:
                break

            pA = p_list[0]
            if len(p_list) >= 2 and not self.solo_mode:
                pB = p_list[1]
                pB.move()
            else:
                pB = self.ai_player
                if self.loaded_model:
                    feat_b = self.get_canonical_features('B')
                    tb = torch.tensor(feat_b, dtype=torch.float32).unsqueeze(0)
                    move_b = self.loaded_model.act(tb, deterministic=False)[0]
                    pB.move(move_b)
                else:
                    pB.move(0)

            pA.tick_cooldown()
            pB.tick_cooldown()
            pA.move()

            if pA.cooldown == 0:
                self.bullets.append(Bullet(pA.x, pA.y, BULLET_SPEED, "A"))
                pA.cooldown = FIRE_COOLDOWN
            if pB.cooldown == 0:
                self.bullets.append(Bullet(pB.x, pB.y, -BULLET_SPEED, "B"))
                pB.cooldown = FIRE_COOLDOWN

            surviving = []
            for b in self.bullets:
                if not b.update():
                    continue

                if b.owner == "B" and math.hypot(b.x - pA.x, b.y - pA.y) < HITBOX_RADIUS:
                    pB.score += 1
                    continue
                if b.owner == "A" and math.hypot(b.x - pB.x, b.y - pB.y) < HITBOX_RADIUS:
                    pA.score += 1
                    continue

                surviving.append(b)

            self.bullets = surviving
            await self.broadcast(self.get_current_state())
            await asyncio.sleep(1.0 / TPS)

lobby = GameLobby()

@app.post("/api/upload_model")
async def upload_model_binary(request: Request, filename: Optional[str] = None):
    try:
        name = filename or request.headers.get("X-Filename") or "uploaded_model.pt"
        name = os.path.basename(name)

        data = await request.body()
        if len(data) == 0:
            return Response(content="Empty file payload.", status_code=400)

        save_path = os.path.join(lobby.models_dir, name)
        with open(save_path, "wb") as f:
            f.write(data)

        lobby.refresh_available_models()
        success, err = lobby.load_ai_model(name)
        if not success:
            return Response(content=f"Saved file, but load failed: {err}", status_code=500)

        await lobby.broadcast({
            "type": "models_updated",
            "models": lobby.available_models,
            "selected_model": name
        })
        await lobby.broadcast(lobby.get_current_state())

        return {"status": "ok", "filename": name, "models": lobby.available_models}
    except Exception as exc:
        return Response(content=f"Server Exception: {exc}", status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_ip = ws.client.host if ws.client else "unknown"

    try:
        init_data = await ws.receive_json()
        username = str(init_data.get("username", "")).strip()[:14] or "Player"
        password = str(init_data.get("password", ""))

        if password != ROOM_PASSWORD:
            await ws.send_json({"type": "error", "message": "Incorrect room password."})
            await ws.close(code=4002)
            return

        for old_ws, old_p in list(lobby.active_connections.items()):
            if old_p.username == username:
                await lobby.disconnect(old_ws)

        if len(lobby.active_connections) >= 2:
            await ws.send_json({"type": "error", "message": "Lobby is full (Maximum 2 players)."})
            await ws.close(code=4001)
            return

        assigned_id = "A" if len(lobby.active_connections) == 0 else "B"
        player = Player(assigned_id, username, 25.0, 20.0 if assigned_id == "A" else 280.0, 
                        0.0 if assigned_id == "A" else 200.0, 
                        99.0 if assigned_id == "A" else 299.0)

        lobby.active_connections[ws] = player
        lobby.refresh_available_models()

        # Send models list & active model immediately
        await ws.send_json({
            "type": "init",
            "slot": assigned_id,
            "username": username,
            "models": lobby.available_models,
            "selected_model": lobby.current_model_name
        })

        await lobby.broadcast(lobby.get_current_state())

        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "ping":
                await ws.send_json({"type": "pong"})

            elif mtype == "select_model":
                m_name = msg.get("model")
                lobby.load_ai_model(m_name)
                await lobby.broadcast({
                    "type": "models_updated",
                    "models": lobby.available_models,
                    "selected_model": lobby.current_model_name
                })
                await lobby.broadcast(lobby.get_current_state())

            elif mtype == "start_match":
                lobby.solo_mode = True
                lobby.running = True
                lobby.reset_arena()
                if lobby.loop_task is None or lobby.loop_task.done():
                    lobby.loop_task = asyncio.create_task(lobby.game_tick())

            elif mtype == "stop_match":
                lobby.running = False
                lobby.reset_arena()
                await lobby.broadcast(lobby.get_current_state())

            elif mtype == "keys":
                player.keys = {
                    "left": bool(msg.get("left", False)),
                    "right": bool(msg.get("right", False)),
                    "up": bool(msg.get("up", False)),
                    "down": bool(msg.get("down", False))
                }

    except WebSocketDisconnect:
        await lobby.disconnect(ws)
    except Exception as e:
        print(f"[Error] {e}")
        await lobby.disconnect(ws)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
