/**
 * Player Controller
 * First-person controls with PointerLock, WASD movement, physics
 */

// Player constants
const PLAYER_HEIGHT = 1.62;
const PLAYER_WIDTH = 0.6;
const PLAYER_EYE_HEIGHT = 1.52;
const WALK_SPEED = 4.3;
const SPRINT_SPEED = 5.6;
const JUMP_VELOCITY = 8.5;
const GRAVITY = 28;
const WATER_GRAVITY = 8;
const WATER_SWIM_SPEED = 2.5;

class Player {
    constructor(camera) {
        this.camera = camera;
        this.position = new THREE.Vector3(0, 80, 0);
        this.velocity = new THREE.Vector3(0, 0, 0);

        // Rotation
        this.pitch = 0;
        this.yaw = 0;

        // State
        this.isGrounded = false;
        this.isSprinting = false;
        this.isInWater = false;
        this.isUnderwater = false;

        // Stats
        this.health = 20;
        this.maxHealth = 20;
        this.hunger = 20;
        this.maxHunger = 20;

        // Input state
        this.keys = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            jump: false,
            sprint: false
        };

        // Pointer lock
        this.isLocked = false;

        // Footstep timer
        this.footstepTimer = 0;
        this.lastFootstepBlock = null;

        // First-person arm and held item
        this.handGroup = null;
        this.heldItemMesh = null;
        this.handSwingTime = 0;
        this.isSwinging = false;
        this.lastHeldType = -1;

        this.initControls();
    }

    initControls() {
        // Keyboard events
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));

        // Mouse events
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mousedown', (e) => this.onMouseDown(e));

        // Pointer lock
        document.addEventListener('pointerlockchange', () => this.onPointerLockChange());

        // Click to play - request pointer lock on canvas click
        const clickToPlay = document.getElementById('click-to-play');
        if (clickToPlay) {
            clickToPlay.addEventListener('click', () => {
                this.requestPointerLock();
                hideClickToPlay();
            });
        }

        // Also allow clicking on game container
        const gameContainer = document.getElementById('game-container');
        if (gameContainer) {
            gameContainer.addEventListener('click', () => {
                if (game && game.isRunning && !this.isLocked && !uiState.isPauseMenuOpen && !uiState.isInventoryOpen) {
                    this.requestPointerLock();
                    hideClickToPlay();
                }
            });
        }
    }

    onKeyDown(e) {
        if (!this.isLocked) return;

        switch (e.code) {
            case 'KeyW':
            case 'ArrowUp':
                this.keys.forward = true;
                break;
            case 'KeyS':
            case 'ArrowDown':
                this.keys.backward = true;
                break;
            case 'KeyA':
            case 'ArrowLeft':
                this.keys.left = true;
                break;
            case 'KeyD':
            case 'ArrowRight':
                this.keys.right = true;
                break;
            case 'Space':
                this.keys.jump = true;
                break;
            case 'ShiftLeft':
            case 'ShiftRight':
                this.keys.sprint = true;
                break;
        }
    }

    onKeyUp(e) {
        switch (e.code) {
            case 'KeyW':
            case 'ArrowUp':
                this.keys.forward = false;
                break;
            case 'KeyS':
            case 'ArrowDown':
                this.keys.backward = false;
                break;
            case 'KeyA':
            case 'ArrowLeft':
                this.keys.left = false;
                break;
            case 'KeyD':
            case 'ArrowRight':
                this.keys.right = false;
                break;
            case 'Space':
                this.keys.jump = false;
                break;
            case 'ShiftLeft':
            case 'ShiftRight':
                this.keys.sprint = false;
                break;
        }
    }

    onMouseMove(e) {
        if (!this.isLocked) return;

        if (this._forceLocked && !e.buttons) {
            // In benchmark mode we treat no-button moves as offsets from screen center,
            // so resetting the cursor to the center does not rotate the camera.
            const centerX = window.innerWidth * 0.5;
            const centerY = window.innerHeight * 0.5;
            const deltaX = Number.isFinite(e.clientX) ? (e.clientX - centerX) : (e.movementX || 0);
            const deltaY = Number.isFinite(e.clientY) ? (e.clientY - centerY) : (e.movementY || 0);
            this._pendingDX = (this._pendingDX || 0) + deltaX;
            this._pendingDY = (this._pendingDY || 0) + deltaY;
            clearTimeout(this._pendingMoveTimer);
            this._pendingMoveTimer = setTimeout(() => {
                this._applyPendingMove();
            }, 10);
            return;
        }

        const sensitivity = 0.002;
        this.yaw -= e.movementX * sensitivity;
        this.pitch -= e.movementY * sensitivity;
        this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
    }

    onMouseDown(e) {
        // Cancel buffered move from click cursor repositioning
        clearTimeout(this._pendingMoveTimer);
        this._pendingDX = 0;
        this._pendingDY = 0;
    }

    _applyPendingMove() {
        const sensitivity = 0.002;
        this.yaw -= (this._pendingDX || 0) * sensitivity;
        this.pitch -= (this._pendingDY || 0) * sensitivity;
        this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
        this._pendingDX = 0;
        this._pendingDY = 0;
    }

    onPointerLockChange() {
        var hasRealLock = document.pointerLockElement !== null;
        if (hasRealLock) {
            this.isLocked = true;
            hideClickToPlay();
        } else if (this._forceLocked) {
            // Benchmark mode: stay "locked" even without real pointer lock
            this.isLocked = true;
        } else if (game && game.isRunning && !game.isPaused && !uiState.isInventoryOpen) {
            this.isLocked = false;
            showPauseMenu();
        } else {
            this.isLocked = false;
        }
    }

    // Force locked state for benchmark (no pointer lock needed)
    forceLock() {
        this._forceLocked = true;
        this.isLocked = true;
        hideClickToPlay();
    }

    requestPointerLock() {
        const canvas = document.querySelector('#game-container canvas');
        if (canvas) {
            canvas.requestPointerLock().catch(e => {
                console.log('Pointer lock request failed:', e.message);
            });
        }
    }

    exitPointerLock() {
        document.exitPointerLock();
    }

    // Get player AABB
    getAABB() {
        const halfWidth = PLAYER_WIDTH / 2;
        return {
            minX: this.position.x - halfWidth,
            maxX: this.position.x + halfWidth,
            minY: this.position.y,
            maxY: this.position.y + PLAYER_HEIGHT,
            minZ: this.position.z - halfWidth,
            maxZ: this.position.z + halfWidth
        };
    }

    // Get block at feet level
    getFeetBlock() {
        return chunkManager.getBlock(
            Math.floor(this.position.x),
            Math.floor(this.position.y),
            Math.floor(this.position.z)
        );
    }

    // Get block at head level
    getHeadBlock() {
        return chunkManager.getBlock(
            Math.floor(this.position.x),
            Math.floor(this.position.y + PLAYER_EYE_HEIGHT),
            Math.floor(this.position.z)
        );
    }

    // Check if player is in water
    checkWater() {
        const feetBlock = this.getFeetBlock();
        const headBlock = this.getHeadBlock();

        this.isInWater = feetBlock === BlockType.WATER || headBlock === BlockType.WATER;
        this.isUnderwater = headBlock === BlockType.WATER;
    }

    // Update physics and movement
    update(deltaTime) {
        if (!this.isLocked) return;

        // Clamp delta time
        deltaTime = Math.min(deltaTime, 0.1);

        // Check water state
        this.checkWater();

        // Calculate movement direction
        const moveDir = new THREE.Vector3();

        if (this.keys.forward) moveDir.z -= 1;
        if (this.keys.backward) moveDir.z += 1;
        if (this.keys.left) moveDir.x -= 1;
        if (this.keys.right) moveDir.x += 1;

        // Normalize
        if (moveDir.length() > 0) {
            moveDir.normalize();
        }

        // Rotate movement by yaw
        // Camera looks along -Z when yaw=0, so forward (moveDir.z=-1) maps to world -Z
        // When yaw rotates, we need: worldX = x*cos + z*sin, worldZ = -x*sin + z*cos
        const sin = Math.sin(this.yaw);
        const cos = Math.cos(this.yaw);
        const moveX = moveDir.x * cos + moveDir.z * sin;
        const moveZ = -moveDir.x * sin + moveDir.z * cos;

        // Determine speed
        let speed = WALK_SPEED;
        if (this.isSprinting && this.keys.sprint && !this.isInWater) {
            speed = SPRINT_SPEED;
        }
        if (this.isInWater) {
            speed = WATER_SWIM_SPEED;
        }

        // Apply movement
        this.velocity.x = moveX * speed;
        this.velocity.z = moveZ * speed;

        // Vertical movement
        if (this.isInWater) {
            // Swimming
            if (this.keys.jump) {
                this.velocity.y = WATER_SWIM_SPEED;
            } else {
                this.velocity.y -= WATER_GRAVITY * deltaTime;
                this.velocity.y = Math.max(this.velocity.y, -WATER_SWIM_SPEED);
            }
        } else {
            // Normal gravity
            this.velocity.y -= GRAVITY * deltaTime;

            // Jump
            if (this.keys.jump && this.isGrounded) {
                this.velocity.y = JUMP_VELOCITY;
                this.isGrounded = false;
                playSound('jump');
            }
        }

        // Apply velocity with collision
        this.moveWithCollision(deltaTime);

        // Update camera
        this.updateCamera();

        // Update hand animation
        this.updateHandAnimation(deltaTime);

        // Footsteps
        this.updateFootsteps(deltaTime);

        // Update sprint state
        this.isSprinting = this.keys.sprint && this.keys.forward && !this.isInWater;
    }

    // Movement with AABB collision
    moveWithCollision(deltaTime) {
        const aabb = this.getAABB();

        // Move X
        this.position.x += this.velocity.x * deltaTime;
        if (this.checkCollision()) {
            this.position.x -= this.velocity.x * deltaTime;
            this.velocity.x = 0;
        }

        // Move Y
        const wasGrounded = this.isGrounded;
        this.position.y += this.velocity.y * deltaTime;

        if (this.checkCollision()) {
            if (this.velocity.y < 0) {
                // Landing
                this.isGrounded = true;
                if (!wasGrounded && !this.isInWater) {
                    playSound('land');
                }
            }
            this.position.y -= this.velocity.y * deltaTime;
            this.velocity.y = 0;
        } else {
            this.isGrounded = false;
        }

        // Move Z
        this.position.z += this.velocity.z * deltaTime;
        if (this.checkCollision()) {
            this.position.z -= this.velocity.z * deltaTime;
            this.velocity.z = 0;
        }

        // Prevent falling through world
        if (this.position.y < 0) {
            this.position.y = 100;
            this.velocity.y = 0;
        }
    }

    // Check collision with world
    checkCollision() {
        const aabb = this.getAABB();

        // Check all blocks that could intersect
        const minBX = Math.floor(aabb.minX);
        const maxBX = Math.floor(aabb.maxX);
        const minBY = Math.floor(aabb.minY);
        const maxBY = Math.floor(aabb.maxY);
        const minBZ = Math.floor(aabb.minZ);
        const maxBZ = Math.floor(aabb.maxZ);

        for (let bx = minBX; bx <= maxBX; bx++) {
            for (let by = minBY; by <= maxBY; by++) {
                for (let bz = minBZ; bz <= maxBZ; bz++) {
                    const block = chunkManager.getBlock(bx, by, bz);
                    if (isBlockSolid(block)) {
                        // Check AABB intersection
                        if (this.aabbIntersectsBlock(aabb, bx, by, bz)) {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    // AABB vs block intersection
    aabbIntersectsBlock(aabb, bx, by, bz) {
        return (
            aabb.minX < bx + 1 &&
            aabb.maxX > bx &&
            aabb.minY < by + 1 &&
            aabb.maxY > by &&
            aabb.minZ < bz + 1 &&
            aabb.maxZ > bz
        );
    }

    // Create first-person hand/arm model
    initHand() {
        this.handGroup = new THREE.Group();

        // Arm (skin-colored box)
        const armGeo = new THREE.BoxGeometry(0.15, 0.15, 0.5);
        const armMat = new THREE.MeshLambertMaterial({ color: 0xd4a574 });
        const arm = new THREE.Mesh(armGeo, armMat);
        arm.position.set(0, -0.05, -0.15);
        this.handGroup.add(arm);

        // Held item placeholder
        this.heldItemMesh = null;

        // Position hand relative to camera
        this.handGroup.position.set(0.35, -0.35, -0.5);
        this.handGroup.rotation.set(0, 0, 0);

        this.camera.add(this.handGroup);
    }

    // Update the held item mesh to match hotbar selection
    updateHeldItem() {
        const selectedSlot = inventory.hotbar[inventory.selectedSlot];
        const blockType = selectedSlot ? selectedSlot.type : -1;

        if (blockType === this.lastHeldType) return;
        this.lastHeldType = blockType;

        // Remove old held item
        if (this.heldItemMesh) {
            this.handGroup.remove(this.heldItemMesh);
            this.heldItemMesh.traverse(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (child.material.map) child.material.map.dispose();
                    child.material.dispose();
                }
            });
            this.heldItemMesh = null;
        }

        if (blockType <= 0) return;

        const icon = getBlockIcon(blockType);
        const tex = new THREE.CanvasTexture(icon);
        tex.flipY = false;
        tex.magFilter = THREE.NearestFilter;
        tex.minFilter = THREE.NearestFilter;

        const props = BlockProperties[blockType];
        const isTool = props && props.isItem && blockType !== BlockType.STICK;
        const isTorch = blockType === BlockType.TORCH;
        const isStick = blockType === BlockType.STICK;

        if (isTool) {
            // 3D tool model
            this.heldItemMesh = this.buildToolMesh(blockType);
            this.heldItemMesh.position.set(0.0, 0.0, -0.35);
            this.heldItemMesh.rotation.set(0.2, 0.3, -0.8);
        } else if (isTorch || isStick) {
            // Stick-like items: thin tall box
            const group = new THREE.Group();
            const stickGeo = new THREE.BoxGeometry(0.04, 0.3, 0.04);
            const stickMat = new THREE.MeshLambertMaterial({ color: 0xa07848 });
            const stick = new THREE.Mesh(stickGeo, stickMat);
            group.add(stick);
            if (isTorch) {
                // Flame on top
                const flameGeo = new THREE.BoxGeometry(0.06, 0.06, 0.06);
                const flameMat = new THREE.MeshBasicMaterial({ color: 0xffcc00 });
                const flame = new THREE.Mesh(flameGeo, flameMat);
                flame.position.set(0, 0.18, 0);
                group.add(flame);
            }
            this.heldItemMesh = group;
            this.heldItemMesh.position.set(0.0, 0.05, -0.3);
            this.heldItemMesh.rotation.set(0.1, 0.3, -0.3);
        } else {
            // Solid blocks: render as small cube with texture
            const size = 0.2;
            const geo = new THREE.BoxGeometry(size, size, size);
            const mat = new THREE.MeshBasicMaterial({ map: tex });
            this.heldItemMesh = new THREE.Mesh(geo, mat);
            this.heldItemMesh.position.set(0, 0.05, -0.3);
            this.heldItemMesh.rotation.set(0.2, 0.5, 0);
        }

        this.handGroup.add(this.heldItemMesh);
    }

    // Build a 3D tool mesh from boxes (Minecraft style)
    buildToolMesh(blockType) {
        const group = new THREE.Group();

        const woodLight = 0xb8945f;
        const woodMid = 0xa07848;
        const woodDark = 0x7a5530;

        // Handle — long thin box
        const handleGeo = new THREE.BoxGeometry(0.04, 0.35, 0.04);
        const handleMat = new THREE.MeshLambertMaterial({ color: woodMid });
        const handle = new THREE.Mesh(handleGeo, handleMat);
        handle.position.set(0, -0.05, 0);
        group.add(handle);

        // Head — varies by tool type
        const headMat = new THREE.MeshLambertMaterial({ color: woodLight });
        const headDarkMat = new THREE.MeshLambertMaterial({ color: woodDark });

        if (blockType === BlockType.WOODEN_PICKAXE) {
            // Horizontal bar
            const barGeo = new THREE.BoxGeometry(0.28, 0.06, 0.06);
            const bar = new THREE.Mesh(barGeo, headMat);
            bar.position.set(0, 0.18, 0);
            group.add(bar);
            // Two down-prongs
            const prongGeo = new THREE.BoxGeometry(0.05, 0.1, 0.05);
            const lp = new THREE.Mesh(prongGeo, headMat);
            lp.position.set(-0.11, 0.11, 0);
            group.add(lp);
            const rp = new THREE.Mesh(prongGeo, headMat);
            rp.position.set(0.11, 0.11, 0);
            group.add(rp);
        } else if (blockType === BlockType.WOODEN_AXE) {
            // Axe blade — flat wide box on one side
            const bladeGeo = new THREE.BoxGeometry(0.16, 0.18, 0.04);
            const blade = new THREE.Mesh(bladeGeo, headMat);
            blade.position.set(-0.1, 0.14, 0);
            group.add(blade);
            // Edge highlight
            const edgeGeo = new THREE.BoxGeometry(0.02, 0.2, 0.05);
            const edge = new THREE.Mesh(edgeGeo, headDarkMat);
            edge.position.set(-0.19, 0.14, 0);
            group.add(edge);
        } else if (blockType === BlockType.WOODEN_SWORD) {
            // Blade — long flat
            const bladeGeo = new THREE.BoxGeometry(0.06, 0.28, 0.03);
            const blade = new THREE.Mesh(bladeGeo, headMat);
            blade.position.set(0, 0.2, 0);
            group.add(blade);
            // Tip
            const tipGeo = new THREE.BoxGeometry(0.04, 0.05, 0.025);
            const tip = new THREE.Mesh(tipGeo, headMat);
            tip.position.set(0, 0.36, 0);
            group.add(tip);
            // Guard
            const guardGeo = new THREE.BoxGeometry(0.14, 0.03, 0.05);
            const guard = new THREE.Mesh(guardGeo, headDarkMat);
            guard.position.set(0, 0.06, 0);
            group.add(guard);
        } else if (blockType === BlockType.WOODEN_SHOVEL) {
            // Flat wide head
            const headGeo = new THREE.BoxGeometry(0.12, 0.14, 0.03);
            const head = new THREE.Mesh(headGeo, headMat);
            head.position.set(0, 0.2, 0);
            group.add(head);
            // Rounded tip (smaller box)
            const tipGeo = new THREE.BoxGeometry(0.08, 0.04, 0.03);
            const tip = new THREE.Mesh(tipGeo, headMat);
            tip.position.set(0, 0.28, 0);
            group.add(tip);
        } else if (blockType === BlockType.WOODEN_HOE) {
            // Horizontal blade on one side
            const bladeGeo = new THREE.BoxGeometry(0.14, 0.06, 0.04);
            const blade = new THREE.Mesh(bladeGeo, headMat);
            blade.position.set(-0.06, 0.18, 0);
            group.add(blade);
            // Single down-tooth
            const toothGeo = new THREE.BoxGeometry(0.05, 0.08, 0.04);
            const tooth = new THREE.Mesh(toothGeo, headMat);
            tooth.position.set(-0.12, 0.11, 0);
            group.add(tooth);
        }

        return group;
    }

    // Update hand swing animation
    updateHandAnimation(deltaTime) {
        if (!this.handGroup) return;

        // Trigger swing on left click
        if (isLeftMouseDown && !this.isSwinging) {
            this.isSwinging = true;
            this.handSwingTime = 0;
        }

        if (this.isSwinging) {
            this.handSwingTime += deltaTime * 6;
            const swing = Math.sin(this.handSwingTime * Math.PI) * 0.6;
            this.handGroup.rotation.x = -swing;

            if (this.handSwingTime >= 1) {
                this.isSwinging = false;
                this.handSwingTime = 0;
                this.handGroup.rotation.x = 0;
            }
        } else {
            // Idle bobbing
            const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.z ** 2);
            if (speed > 0.5 && this.isGrounded) {
                const bob = Math.sin(Date.now() * 0.008) * 0.02;
                const sway = Math.cos(Date.now() * 0.004) * 0.01;
                this.handGroup.position.y = -0.35 + bob;
                this.handGroup.position.x = 0.35 + sway;
            } else {
                this.handGroup.position.y = -0.35;
                this.handGroup.position.x = 0.35;
            }
        }

        // Update held item to match selected slot
        this.updateHeldItem();
    }

    // Update camera position and rotation
    updateCamera() {
        // Position at eye level
        this.camera.position.set(
            this.position.x,
            this.position.y + PLAYER_EYE_HEIGHT,
            this.position.z
        );

        // Apply rotation
        this.camera.rotation.order = 'YXZ';
        this.camera.rotation.y = this.yaw;
        this.camera.rotation.x = this.pitch;
    }

    // Footstep sounds
    updateFootsteps(deltaTime) {
        if (!this.isGrounded || this.isInWater) {
            this.footstepTimer = 0;
            return;
        }

        const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.z ** 2);
        if (speed < 0.1) {
            this.footstepTimer = 0;
            return;
        }

        const interval = this.isSprinting ? 0.3 : 0.4;
        this.footstepTimer += deltaTime;

        if (this.footstepTimer >= interval) {
            this.footstepTimer = 0;

            // Get block below feet
            const block = chunkManager.getBlock(
                Math.floor(this.position.x),
                Math.floor(this.position.y - 0.1),
                Math.floor(this.position.z)
            );

            if (block !== this.lastFootstepBlock || block !== BlockType.AIR) {
                playSound('footstep', block);
                this.lastFootstepBlock = block;

                // Spawn dust particles
                if (block !== BlockType.WATER) {
                    spawnDustParticles(
                        this.position.x,
                        this.position.y,
                        this.position.z
                    );
                }
            }
        }
    }

    // Get look direction
    getLookDirection() {
        const dir = new THREE.Vector3(0, 0, -1);
        dir.applyEuler(new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ'));
        return dir;
    }

    // Take damage
    takeDamage(amount) {
        this.health -= amount;
        playSound('hurt');

        if (this.health <= 0) {
            this.die();
        }

        updateHealthDisplay();
    }

    // Heal
    heal(amount) {
        this.health = Math.min(this.maxHealth, this.health + amount);
        updateHealthDisplay();
    }

    // Die
    die() {
        this.health = 0;
        // Respawn
        setTimeout(() => {
            this.respawn();
        }, 2000);
    }

    // Respawn
    respawn() {
        const spawn = getSpawnPosition(chunkManager);
        this.position.set(spawn.x, spawn.y, spawn.z);
        this.velocity.set(0, 0, 0);
        this.health = this.maxHealth;
        this.hunger = this.maxHunger;
        updateHealthDisplay();
        updateHungerDisplay();
    }

    // Get position for saving
    getPosition() {
        return {
            x: this.position.x,
            y: this.position.y,
            z: this.position.z,
            yaw: this.yaw,
            pitch: this.pitch
        };
    }

    // Set position from save
    setPosition(pos) {
        this.position.set(pos.x, pos.y, pos.z);
        this.yaw = pos.yaw || 0;
        this.pitch = pos.pitch || 0;
    }
}

// Global player instance
let player = null;

function initPlayer(camera) {
    player = new Player(camera);
    player.initHand();
    return player;
}
