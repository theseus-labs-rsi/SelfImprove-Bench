/**
 * Particle System
 * Block break particles, walking dust
 */

// Particle pool for performance
const MAX_PARTICLES = 500;

// Particle system
class ParticleSystem {
    constructor(scene) {
        this.scene = scene;
        this.particles = [];
        this.mesh = null;

        this.initMesh();
    }

    initMesh() {
        // Create instanced mesh for particles
        const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });

        // Use regular meshes pooled
        this.particleMeshes = [];
        this.availableParticles = [];

        for (let i = 0; i < MAX_PARTICLES; i++) {
            const mesh = new THREE.Mesh(geometry.clone(), material.clone());
            mesh.visible = false;
            mesh.frustumCulled = false;
            this.scene.add(mesh);

            this.particleMeshes.push({
                mesh: mesh,
                active: false,
                velocity: new THREE.Vector3(),
                life: 0,
                maxLife: 0
            });

            this.availableParticles.push(i);
        }
    }

    // Spawn block break particles
    spawnBreakParticles(x, y, z, blockType, count = 10) {
        // Get block color
        const color = this.getBlockColor(blockType);

        for (let i = 0; i < count; i++) {
            const particle = this.getParticle();
            if (!particle) break;

            const mesh = particle.mesh;

            // Random position within block
            mesh.position.set(
                x + (Math.random() - 0.5) * 0.8,
                y + (Math.random() - 0.5) * 0.8,
                z + (Math.random() - 0.5) * 0.8
            );

            // Random velocity (explosive)
            particle.velocity.set(
                (Math.random() - 0.5) * 3,
                Math.random() * 4,
                (Math.random() - 0.5) * 3
            );

            // Set color with variation
            const colorVariation = 0.8 + Math.random() * 0.4;
            mesh.material.color.setHex(color);
            mesh.material.color.multiplyScalar(colorVariation);

            // Scale variation
            const scale = 0.05 + Math.random() * 0.1;
            mesh.scale.setScalar(scale);

            particle.life = 0;
            particle.maxLife = 0.5 + Math.random() * 0.5;
            particle.active = true;
            mesh.visible = true;
        }
    }

    // Spawn walking dust
    spawnDustParticles(x, y, z) {
        const particle = this.getParticle();
        if (!particle) return;

        const mesh = particle.mesh;

        // Position at feet
        mesh.position.set(
            x + (Math.random() - 0.5) * 0.3,
            y,
            z + (Math.random() - 0.5) * 0.3
        );

        // Small upward velocity
        particle.velocity.set(
            (Math.random() - 0.5) * 0.5,
            Math.random() * 0.5,
            (Math.random() - 0.5) * 0.5
        );

        // Gray dust color
        mesh.material.color.setHex(0x888888);
        mesh.material.opacity = 0.5;

        const scale = 0.03 + Math.random() * 0.03;
        mesh.scale.setScalar(scale);

        particle.life = 0;
        particle.maxLife = 0.3 + Math.random() * 0.2;
        particle.active = true;
        mesh.visible = true;
    }

    getParticle() {
        if (this.availableParticles.length === 0) return null;
        const index = this.availableParticles.pop();
        return this.particleMeshes[index];
    }

    returnParticle(particleIndex) {
        this.availableParticles.push(particleIndex);
    }

    // Get color for block type
    getBlockColor(blockType) {
        const colors = {
            [BlockType.GRASS]: 0x5d9b3d,
            [BlockType.DIRT]: 0x8b6b4a,
            [BlockType.STONE]: 0x7a7a7a,
            [BlockType.WOOD_LOG]: 0x6b5030,
            [BlockType.LEAVES]: 0x3a8b2a,
            [BlockType.SAND]: 0xe8d9a0,
            [BlockType.WATER]: 0x1e64b4,
            [BlockType.BEDROCK]: 0x333333,
            [BlockType.WOOD_PLANK]: 0xb8945f,
            [BlockType.COBBLESTONE]: 0x6a6a6a,
            [BlockType.COAL_ORE]: 0x2a2a2a,
            [BlockType.IRON_ORE]: 0xc9a080,
            [BlockType.GOLD_ORE]: 0xfcee4b,
            [BlockType.DIAMOND_ORE]: 0x5decf5,
            [BlockType.GLASS]: 0xc8e6ff,
            [BlockType.CRAFTING_TABLE]: 0xb8945f,
            [BlockType.SNOW]: 0xf0f0f0,
            [BlockType.CACTUS]: 0x2a8b2a
        };

        return colors[blockType] || 0x808080;
    }

    // Update all particles
    update(deltaTime) {
        for (let i = 0; i < this.particleMeshes.length; i++) {
            const particle = this.particleMeshes[i];
            if (!particle.active) continue;

            particle.life += deltaTime;

            if (particle.life >= particle.maxLife) {
                particle.active = false;
                particle.mesh.visible = false;
                this.returnParticle(i);
                continue;
            }

            // Apply gravity
            particle.velocity.y -= 15 * deltaTime;

            // Update position
            particle.mesh.position.add(
                particle.velocity.clone().multiplyScalar(deltaTime)
            );

            // Fade out
            const lifeRatio = particle.life / particle.maxLife;
            particle.mesh.material.opacity = 1 - lifeRatio;

            // Shrink
            const scale = particle.mesh.scale.x * (1 - lifeRatio * 0.5);
            particle.mesh.scale.setScalar(Math.max(0.01, scale));

            // Ground collision
            const groundY = this.getGroundLevel(
                particle.mesh.position.x,
                particle.mesh.position.z
            );

            if (particle.mesh.position.y < groundY) {
                particle.mesh.position.y = groundY;
                particle.velocity.y *= -0.3;
                particle.velocity.x *= 0.8;
                particle.velocity.z *= 0.8;
            }
        }
    }

    // Get ground level at position
    getGroundLevel(x, z) {
        // Search from a reasonable height around the particle
        const startY = Math.min(CHUNK_HEIGHT - 1, 128);
        for (let y = startY; y >= 0; y--) {
            const block = chunkManager.getBlock(
                Math.floor(x),
                y,
                Math.floor(z)
            );
            if (isBlockSolid(block)) {
                return y + 1;
            }
        }
        return 0;
    }

    // Clear all particles
    clear() {
        for (const particle of this.particleMeshes) {
            particle.active = false;
            particle.mesh.visible = false;
        }
        this.availableParticles = [];
        for (let i = 0; i < MAX_PARTICLES; i++) {
            this.availableParticles.push(i);
        }
    }
}

// Global functions for easy access
let particleSystem = null;

function initParticles(scene) {
    particleSystem = new ParticleSystem(scene);
    return particleSystem;
}

function spawnBreakParticles(x, y, z, blockType, count) {
    if (particleSystem) {
        particleSystem.spawnBreakParticles(x, y, z, blockType, count);
    }
}

function spawnDustParticles(x, y, z) {
    if (particleSystem) {
        particleSystem.spawnDustParticles(x, y, z);
    }
}

function updateParticles(deltaTime) {
    if (particleSystem) {
        particleSystem.update(deltaTime);
    }
}

function clearParticles() {
    if (particleSystem) {
        particleSystem.clear();
    }
}
