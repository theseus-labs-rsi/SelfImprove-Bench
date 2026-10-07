/**
 * Mob System
 * Passive (Pig) and Hostile (Zombie) mobs
 */

// Mob types
const MobType = {
    PIG: 'pig',
    ZOMBIE: 'zombie'
};

// Mob properties
const MobProperties = {
    [MobType.PIG]: {
        health: 10,
        speed: 1.0,
        hostile: false,
        spawnDay: true,
        spawnNight: true,
        dropItems: [BlockType.DIRT] // Simplified
    },
    [MobType.ZOMBIE]: {
        health: 20,
        speed: 0.8,
        hostile: true,
        spawnDay: false,
        spawnNight: true,
        damage: 3,
        dropItems: [BlockType.COBBLESTONE] // Simplified
    }
};

// Mob class
class Mob {
    constructor(type, x, y, z) {
        this.type = type;
        this.props = MobProperties[type];

        this.position = new THREE.Vector3(x, y, z);
        this.velocity = new THREE.Vector3(0, 0, 0);
        this.rotation = Math.random() * Math.PI * 2;

        this.health = this.props.health;
        this.maxHealth = this.props.health;

        this.isGrounded = false;
        this.targetPlayer = false;
        this.wanderAngle = Math.random() * Math.PI * 2;
        this.wanderTimer = 0;
        this.attackCooldown = 0;

        this.mesh = null;
        this.animationTime = 0;

        this.createMesh();
    }

    createMesh() {
        const group = new THREE.Group();

        // Body dimensions based on type
        let bodyW, bodyH, bodyD;
        let headW, headH, headD;
        let legW, legH;

        if (this.type === MobType.PIG) {
            bodyW = 0.6; bodyH = 0.5; bodyD = 1.0;
            headW = 0.5; headH = 0.5; headD = 0.5;
            legW = 0.2; legH = 0.3;
        } else {
            // Zombie - more humanoid
            bodyW = 0.5; bodyH = 0.75; bodyD = 0.3;
            headW = 0.5; headH = 0.5; headD = 0.5;
            legW = 0.25; legH = 0.75;
        }

        // Create textures
        const bodyMat = this.createMaterial('body');
        const headMat = this.createMaterial('head');
        const legMat = this.createMaterial('leg');

        // Body
        const bodyGeo = new THREE.BoxGeometry(bodyW, bodyH, bodyD);
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.y = legH + bodyH / 2;
        group.add(body);
        this.bodyMesh = body;

        // Head
        const headGeo = new THREE.BoxGeometry(headW, headH, headD);
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.y = legH + bodyH + headH / 2;
        head.position.z = bodyD / 2;
        group.add(head);
        this.headMesh = head;

        // Legs
        this.legs = [];
        const legGeo = new THREE.BoxGeometry(legW, legH, legW);

        const legPositions = [
            [-bodyW / 2 + legW / 2, legH / 2, -bodyD / 2 + legW / 2 + 0.1],
            [bodyW / 2 - legW / 2, legH / 2, -bodyD / 2 + legW / 2 + 0.1],
            [-bodyW / 2 + legW / 2, legH / 2, bodyD / 2 - legW / 2 - 0.1],
            [bodyW / 2 - legW / 2, legH / 2, bodyD / 2 - legW / 2 - 0.1]
        ];

        for (const pos of legPositions) {
            const leg = new THREE.Mesh(legGeo, legMat);
            leg.position.set(pos[0], pos[1], pos[2]);
            group.add(leg);
            this.legs.push(leg);
        }

        // Arms for zombie
        if (this.type === MobType.ZOMBIE) {
            const armGeo = new THREE.BoxGeometry(0.25, 0.6, 0.25);
            const armMat = this.createMaterial('arm');

            const leftArm = new THREE.Mesh(armGeo, armMat);
            leftArm.position.set(-bodyW / 2 - 0.125, legH + bodyH / 2, bodyD / 2);
            leftArm.rotation.x = Math.PI / 2;
            group.add(leftArm);
            this.leftArm = leftArm;

            const rightArm = new THREE.Mesh(armGeo, armMat);
            rightArm.position.set(bodyW / 2 + 0.125, legH + bodyH / 2, bodyD / 2);
            rightArm.rotation.x = Math.PI / 2;
            group.add(rightArm);
            this.rightArm = rightArm;
        }

        group.position.copy(this.position);
        this.mesh = group;
    }

    createMaterial(part) {
        // Create procedural texture for mob
        const canvas = document.createElement('canvas');
        canvas.width = 16;
        canvas.height = 16;
        const ctx = canvas.getContext('2d');

        if (this.type === MobType.PIG) {
            // Pink pig
            ctx.fillStyle = '#f0a0a0';
            ctx.fillRect(0, 0, 16, 16);

            // Darker details
            ctx.fillStyle = '#d08080';
            ctx.fillRect(0, 0, 16, 2);
            ctx.fillRect(0, 14, 16, 2);

            // Eyes
            if (part === 'head') {
                ctx.fillStyle = '#000';
                ctx.fillRect(4, 5, 2, 2);
                ctx.fillRect(10, 5, 2, 2);
                // Snout
                ctx.fillStyle = '#e09090';
                ctx.fillRect(5, 9, 6, 4);
            }
        } else {
            // Zombie green
            ctx.fillStyle = '#5a8a5a';
            ctx.fillRect(0, 0, 16, 16);

            // Tattered clothes look
            ctx.fillStyle = '#4a6a4a';
            for (let i = 0; i < 10; i++) {
                ctx.fillRect(Math.random() * 14, Math.random() * 14, 2, 2);
            }

            // Face details
            if (part === 'head') {
                // Eyes
                ctx.fillStyle = '#000';
                ctx.fillRect(3, 4, 3, 2);
                ctx.fillRect(10, 4, 3, 2);
                // Mouth
                ctx.fillStyle = '#2a2a2a';
                ctx.fillRect(4, 10, 8, 3);
            }
        }

        const texture = new THREE.CanvasTexture(canvas);
        texture.magFilter = THREE.NearestFilter;
        texture.minFilter = THREE.NearestFilter;

        return new THREE.MeshLambertMaterial({ map: texture });
    }

    update(deltaTime, player, dayTime) {
        this.animationTime += deltaTime;

        // AI behavior
        if (this.props.hostile) {
            this.updateHostileAI(deltaTime, player);
        } else {
            this.updatePassiveAI(deltaTime);
        }

        // Apply physics
        this.applyPhysics(deltaTime);

        // Update mesh position
        this.mesh.position.copy(this.position);
        this.mesh.rotation.y = this.rotation;

        // Animate
        this.animate(deltaTime);

        // Attack cooldown
        if (this.attackCooldown > 0) {
            this.attackCooldown -= deltaTime;
        }
    }

    updatePassiveAI(deltaTime) {
        // Random wandering
        this.wanderTimer -= deltaTime;

        if (this.wanderTimer <= 0) {
            this.wanderAngle = Math.random() * Math.PI * 2;
            this.wanderTimer = 2 + Math.random() * 4;
        }

        // Move in wander direction
        const speed = this.props.speed * 0.5;
        this.velocity.x = Math.sin(this.wanderAngle) * speed;
        this.velocity.z = Math.cos(this.wanderAngle) * speed;

        // Face movement direction
        this.rotation = this.wanderAngle;

        // Random jump
        if (this.isGrounded && Math.random() < 0.01) {
            this.velocity.y = 5;
        }
    }

    updateHostileAI(deltaTime, target) {
        if (!target) return;

        const dx = target.position.x - this.position.x;
        const dz = target.position.z - this.position.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        // Check if can see player (simplified - always sees at night)
        if (dist < 20) {
            // Face player
            this.rotation = Math.atan2(dx, dz);

            // Move towards player
            const speed = this.props.speed;
            this.velocity.x = (dx / dist) * speed;
            this.velocity.z = (dz / dist) * speed;

            // Jump if blocked
            if (this.isGrounded) {
                const frontBlock = chunkManager.getBlock(
                    Math.floor(this.position.x + Math.sin(this.rotation)),
                    Math.floor(this.position.y + 1),
                    Math.floor(this.position.z + Math.cos(this.rotation))
                );

                if (isBlockSolid(frontBlock)) {
                    this.velocity.y = 6;
                }
            }

            // Attack player if close
            if (dist < 1.5 && this.attackCooldown <= 0) {
                this.attackPlayer(target);
            }
        } else {
            // Wander when player is far
            this.updatePassiveAI(deltaTime);
        }
    }

    attackPlayer(target) {
        target.takeDamage(this.props.damage || 1);
        this.attackCooldown = 1.0;

        // Knockback
        const dx = target.position.x - this.position.x;
        const dz = target.position.z - this.position.z;
        const dist = Math.sqrt(dx * dx + dz * dz);

        target.velocity.x += (dx / dist) * 5;
        target.velocity.z += (dz / dist) * 5;
        target.velocity.y = 3;
    }

    applyPhysics(deltaTime) {
        // Gravity
        this.velocity.y -= 20 * deltaTime;

        // Move with collision
        this.position.x += this.velocity.x * deltaTime;
        if (this.checkCollision()) {
            this.position.x -= this.velocity.x * deltaTime;
            this.velocity.x = 0;
        }

        this.position.y += this.velocity.y * deltaTime;
        if (this.checkCollision()) {
            if (this.velocity.y < 0) {
                this.isGrounded = true;
            }
            this.position.y -= this.velocity.y * deltaTime;
            this.velocity.y = 0;
        } else {
            this.isGrounded = false;
        }

        this.position.z += this.velocity.z * deltaTime;
        if (this.checkCollision()) {
            this.position.z -= this.velocity.z * deltaTime;
            this.velocity.z = 0;
        }

        // Friction
        this.velocity.x *= 0.8;
        this.velocity.z *= 0.8;
    }

    checkCollision() {
        const hw = 0.3;
        const h = 1.0;

        const minBX = Math.floor(this.position.x - hw);
        const maxBX = Math.floor(this.position.x + hw);
        const minBY = Math.floor(this.position.y);
        const maxBY = Math.floor(this.position.y + h);
        const minBZ = Math.floor(this.position.z - hw);
        const maxBZ = Math.floor(this.position.z + hw);

        for (let bx = minBX; bx <= maxBX; bx++) {
            for (let by = minBY; by <= maxBY; by++) {
                for (let bz = minBZ; bz <= maxBZ; bz++) {
                    const block = chunkManager.getBlock(bx, by, bz);
                    if (isBlockSolid(block)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    animate(deltaTime) {
        // Leg walking animation
        const walkSpeed = 8;
        const walkAmount = 0.5;

        const isMoving = Math.abs(this.velocity.x) > 0.1 || Math.abs(this.velocity.z) > 0.1;

        if (isMoving && this.legs) {
            const time = this.animationTime * walkSpeed;

            this.legs[0].rotation.x = Math.sin(time) * walkAmount;
            this.legs[1].rotation.x = -Math.sin(time) * walkAmount;
            this.legs[2].rotation.x = -Math.sin(time) * walkAmount;
            this.legs[3].rotation.x = Math.sin(time) * walkAmount;

            // Arm swing for zombie
            if (this.leftArm && this.rightArm) {
                this.leftArm.rotation.x = Math.PI / 2 + Math.sin(time) * 0.3;
                this.rightArm.rotation.x = Math.PI / 2 - Math.sin(time) * 0.3;
            }
        } else {
            // Reset legs
            this.legs.forEach(leg => leg.rotation.x = 0);
            if (this.leftArm) this.leftArm.rotation.x = Math.PI / 2;
            if (this.rightArm) this.rightArm.rotation.x = Math.PI / 2;
        }
    }

    takeDamage(amount) {
        this.health -= amount;

        // Flash red
        this.mesh.traverse(child => {
            if (child.material) {
                child.material.emissive = new THREE.Color(0xff0000);
                setTimeout(() => {
                    if (child.material) {
                        child.material.emissive = new THREE.Color(0x000000);
                    }
                }, 100);
            }
        });

        playSound('mobHurt');

        if (this.health <= 0) {
            this.die();
        }
    }

    die() {
        // Drop items
        const drops = this.props.dropItems;
        if (drops && drops.length > 0) {
            const drop = drops[Math.floor(Math.random() * drops.length)];
            addToInventory(drop, 1);
        }

        // Death sound
        playSound('mobDeath');

        // Remove from scene
        if (this.mesh.parent) {
            this.mesh.parent.remove(this.mesh);
        }

        // Mark for removal
        this.isDead = true;
    }

    dispose() {
        this.mesh.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        });
    }
}

// Mob manager
class MobManager {
    constructor(scene) {
        this.scene = scene;
        this.mobs = [];
        this.maxMobs = 20;
        this.spawnCooldown = 0;
    }

    update(deltaTime, player, dayTime) {
        // Spawn mobs
        this.spawnCooldown -= deltaTime;
        if (this.spawnCooldown <= 0 && this.mobs.length < this.maxMobs) {
            this.trySpawnMob(player, dayTime);
            this.spawnCooldown = 5;
        }

        // Update all mobs
        for (let i = this.mobs.length - 1; i >= 0; i--) {
            const mob = this.mobs[i];

            if (mob.isDead) {
                mob.dispose();
                this.mobs.splice(i, 1);
                continue;
            }

            // Despawn if too far from player
            const dist = mob.position.distanceTo(player.position);
            if (dist > 100) {
                this.scene.remove(mob.mesh);
                mob.dispose();
                this.mobs.splice(i, 1);
                continue;
            }

            mob.update(deltaTime, player, dayTime);
        }
    }

    trySpawnMob(player, dayTime) {
        // Random position around player
        const angle = Math.random() * Math.PI * 2;
        const dist = 20 + Math.random() * 20;
        const spawnX = player.position.x + Math.cos(angle) * dist;
        const spawnZ = player.position.z + Math.sin(angle) * dist;

        // Find ground level
        let spawnY = -1;
        for (let y = CHUNK_HEIGHT - 1; y >= 0; y--) {
            const block = chunkManager.getBlock(
                Math.floor(spawnX),
                y,
                Math.floor(spawnZ)
            );
            if (isBlockSolid(block)) {
                spawnY = y + 1;
                break;
            }
        }

        if (spawnY < 0) return;

        // Check light level (simplified)
        const isNight = dayTime > 0.75 || dayTime < 0.25;

        // Determine mob type
        let mobType;
        if (isNight) {
            // More hostile mobs at night
            mobType = Math.random() < 0.7 ? MobType.ZOMBIE : MobType.PIG;
        } else {
            // Only passive mobs during day
            mobType = MobType.PIG;
        }

        // Create mob
        const mob = new Mob(mobType, spawnX, spawnY, spawnZ);
        this.mobs.push(mob);
        this.scene.add(mob.mesh);
    }

    getMobs() {
        return this.mobs;
    }

    clear() {
        for (const mob of this.mobs) {
            this.scene.remove(mob.mesh);
            mob.dispose();
        }
        this.mobs = [];
    }
}

// Global mob manager
let mobManager = null;

function initMobManager(scene) {
    mobManager = new MobManager(scene);
    return mobManager;
}
