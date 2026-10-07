/**
 * Chunk System with Greedy Meshing
 * Chunks are 16x256x16 blocks
 */

const CHUNK_SIZE = 16;
const CHUNK_HEIGHT = 256;
const RENDER_DISTANCE = 6;

// Chunk data structure
class Chunk {
    constructor(cx, cz) {
        this.cx = cx;
        this.cz = cz;
        this.blocks = new Uint8Array(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE);
        this.mesh = null;
        this.waterMesh = null;
        this.needsUpdate = true;
        this.isGenerated = false;
    }

    // Get block at local coordinates
    getBlock(lx, ly, lz) {
        if (lx < 0 || lx >= CHUNK_SIZE || ly < 0 || ly >= CHUNK_HEIGHT || lz < 0 || lz >= CHUNK_SIZE) {
            return BlockType.AIR;
        }
        return this.blocks[lx + lz * CHUNK_SIZE + ly * CHUNK_SIZE * CHUNK_SIZE];
    }

    // Set block at local coordinates
    setBlock(lx, ly, lz, blockType) {
        if (lx < 0 || lx >= CHUNK_SIZE || ly < 0 || ly >= CHUNK_HEIGHT || lz < 0 || lz >= CHUNK_SIZE) {
            return false;
        }
        this.blocks[lx + lz * CHUNK_SIZE + ly * CHUNK_SIZE * CHUNK_SIZE] = blockType;
        this.needsUpdate = true;
        return true;
    }

    // Get world coordinates
    getWorldX(lx) {
        return this.cx * CHUNK_SIZE + lx;
    }

    getWorldZ(lz) {
        return this.cz * CHUNK_SIZE + lz;
    }
}

// Chunk manager
class ChunkManager {
    constructor(scene) {
        this.scene = scene;
        this.chunks = new Map();
        this.chunkMeshes = new Map();
        this.modifications = new Map(); // Player modifications for saving
    }

    // Get chunk key
    getChunkKey(cx, cz) {
        return `${cx},${cz}`;
    }

    // Get chunk at chunk coordinates
    getChunk(cx, cz) {
        return this.chunks.get(this.getChunkKey(cx, cz));
    }

    // Get chunk at world coordinates
    getChunkAtWorld(wx, wz) {
        const cx = Math.floor(wx / CHUNK_SIZE);
        const cz = Math.floor(wz / CHUNK_SIZE);
        return this.getChunk(cx, cz);
    }

    // Create or get chunk
    createChunk(cx, cz) {
        const key = this.getChunkKey(cx, cz);
        if (!this.chunks.has(key)) {
            const chunk = new Chunk(cx, cz);
            this.chunks.set(key, chunk);
            return chunk;
        }
        return this.chunks.get(key);
    }

    // Get block at world coordinates
    getBlock(wx, wy, wz) {
        const cx = Math.floor(wx / CHUNK_SIZE);
        const cz = Math.floor(wz / CHUNK_SIZE);
        const chunk = this.getChunk(cx, cz);
        if (!chunk) return BlockType.AIR;

        const lx = ((wx % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
        const lz = ((wz % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
        return chunk.getBlock(lx, wy, lz);
    }

    // Set block at world coordinates
    setBlock(wx, wy, wz, blockType) {
        const cx = Math.floor(wx / CHUNK_SIZE);
        const cz = Math.floor(wz / CHUNK_SIZE);
        const chunk = this.getChunk(cx, cz);
        if (!chunk) return false;

        const lx = ((wx % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
        const lz = ((wz % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;

        if (chunk.setBlock(lx, wy, lz, blockType)) {
            // Save modification
            const modKey = `${wx},${wy},${wz}`;
            if (blockType === BlockType.AIR) {
                this.modifications.set(modKey, -1); // -1 means removed
            } else {
                this.modifications.set(modKey, blockType);
            }

            // Mark neighboring chunks for update if on edge
            if (lx === 0) this.markChunkForUpdate(cx - 1, cz);
            if (lx === CHUNK_SIZE - 1) this.markChunkForUpdate(cx + 1, cz);
            if (lz === 0) this.markChunkForUpdate(cx, cz - 1);
            if (lz === CHUNK_SIZE - 1) this.markChunkForUpdate(cx, cz + 1);

            return true;
        }
        return false;
    }

    markChunkForUpdate(cx, cz) {
        const chunk = this.getChunk(cx, cz);
        if (chunk) chunk.needsUpdate = true;
    }

    // Apply saved modifications
    applyModifications(modifications) {
        for (const [key, value] of Object.entries(modifications)) {
            const [wx, wy, wz] = key.split(',').map(Number);
            const cx = Math.floor(wx / CHUNK_SIZE);
            const cz = Math.floor(wz / CHUNK_SIZE);
            const chunk = this.getChunk(cx, cz);

            if (chunk) {
                const lx = ((wx % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
                const lz = ((wz % CHUNK_SIZE) + CHUNK_SIZE) % CHUNK_SIZE;
                const blockType = value === -1 ? BlockType.AIR : value;
                chunk.setBlock(lx, wy, lz, blockType);
            }
        }
        this.modifications = new Map(Object.entries(modifications));
    }

    // Build chunk mesh with greedy meshing
    buildChunkMesh(chunk) {
        if (chunk.mesh) {
            this.scene.remove(chunk.mesh);
            chunk.mesh.geometry.dispose();
        }
        if (chunk.waterMesh) {
            this.scene.remove(chunk.waterMesh);
            chunk.waterMesh.geometry.dispose();
        }
        if (chunk.spriteMesh) {
            this.scene.remove(chunk.spriteMesh);
            chunk.spriteMesh.geometry.dispose();
        }

        const vertices = [];
        const uvs = [];
        const normals = [];
        const colors = [];

        const waterVertices = [];
        const waterUvs = [];
        const waterNormals = [];
        const waterColors = [];

        // Face directions: right, left, top, bottom, front, back
        // ORIGINAL corner ordering preserved for correct triangle winding (front-face culling)
        const faces = [
            { dir: [1, 0, 0], corners: [[1,0,0], [1,1,0], [1,1,1], [1,0,1]], normal: [1,0,0] },
            { dir: [-1, 0, 0], corners: [[0,0,1], [0,1,1], [0,1,0], [0,0,0]], normal: [-1,0,0] },
            { dir: [0, 1, 0], corners: [[0,1,1], [1,1,1], [1,1,0], [0,1,0]], normal: [0,1,0] },
            { dir: [0, -1, 0], corners: [[0,0,0], [1,0,0], [1,0,1], [0,0,1]], normal: [0,-1,0] },
            { dir: [0, 0, 1], corners: [[0,0,1], [1,0,1], [1,1,1], [0,1,1]], normal: [0,0,1] },
            { dir: [0, 0, -1], corners: [[1,0,0], [0,0,0], [0,1,0], [1,1,0]], normal: [0,0,-1] }
        ];

        const faceNames = ['side', 'side', 'top', 'bottom', 'side', 'side'];

        // Per-face UV mappings ensuring texture v=0 (top/green) aligns with block y=1
        // right/left: corners naturally go y=0→y=1→y=1→y=0
        // front/back: corners go y=0→y=0→y=1→y=1 (different order, need different UVs)
        const faceUvs = [
            [[0,1], [0,0], [1,0], [1,1]], // right (+X)
            [[0,1], [0,0], [1,0], [1,1]], // left (-X)
            [[0,1], [0,0], [1,0], [1,1]], // top (+Y)
            [[0,1], [0,0], [1,0], [1,1]], // bottom (-Y)
            [[0,1], [1,1], [1,0], [0,0]], // front (+Z): c0=BL c1=BR c2=TR c3=TL
            [[0,1], [1,1], [1,0], [0,0]], // back (-Z):  c0=BL c1=BR c2=TR c3=TL
        ];

        // Sprite-style blocks rendered as two crossed planes
        const spriteVertices = [];
        const spriteUvs = [];
        const spriteNormals = [];
        const spriteColors = [];

        // Iterate through all blocks
        for (let y = 0; y < CHUNK_HEIGHT; y++) {
            for (let z = 0; z < CHUNK_SIZE; z++) {
                for (let x = 0; x < CHUNK_SIZE; x++) {
                    const blockType = chunk.getBlock(x, y, z);
                    if (blockType === BlockType.AIR) continue;

                    const wx = chunk.getWorldX(x);
                    const wz = chunk.getWorldZ(z);

                    // Torch: render as crossed planes (X shape)
                    if (blockType === BlockType.TORCH) {
                        const uv = getBlockUVs(blockType, 'all');
                        // Two crossed quads forming an X
                        const crossFaces = [
                            // Plane 1: diagonal from (0,0,0) to (1,0,1)
                            { corners: [[0,0,0],[0,1,0],[1,1,1],[1,0,1]], normal: [-0.707,0,0.707] },
                            { corners: [[1,0,1],[1,1,1],[0,1,0],[0,0,0]], normal: [0.707,0,-0.707] },
                            // Plane 2: diagonal from (1,0,0) to (0,0,1)
                            { corners: [[1,0,0],[1,1,0],[0,1,1],[0,0,1]], normal: [0.707,0,0.707] },
                            { corners: [[0,0,1],[0,1,1],[1,1,0],[1,0,0]], normal: [-0.707,0,-0.707] },
                        ];
                        for (const cf of crossFaces) {
                            const indices = [0, 1, 2, 0, 2, 3];
                            for (const i of indices) {
                                const c = cf.corners[i];
                                spriteVertices.push(wx + c[0], y + c[1], wz + c[2]);
                                const uvCorner = [[0,1],[0,0],[1,0],[1,1]][i];
                                spriteUvs.push(
                                    uv.u0 + uvCorner[0] * (uv.u1 - uv.u0),
                                    uv.v0 + uvCorner[1] * (uv.v1 - uv.v0)
                                );
                                spriteNormals.push(cf.normal[0], cf.normal[1], cf.normal[2]);
                                spriteColors.push(1.0, 1.0, 1.0);
                            }
                        }
                        continue;
                    }

                    const isWater = blockType === BlockType.WATER;
                    const targetVerts = isWater ? waterVertices : vertices;
                    const targetUvs = isWater ? waterUvs : uvs;
                    const targetNormals = isWater ? waterNormals : normals;
                    const targetColors = isWater ? waterColors : colors;

                    // Check each face
                    for (let f = 0; f < faces.length; f++) {
                        const face = faces[f];
                        const [dx, dy, dz] = face.dir;

                        // Get neighbor block
                        const neighborType = this.getBlock(wx + dx, y + dy, wz + dz);

                        // Only render face if neighbor is transparent
                        if (!isBlockTransparent(neighborType)) continue;

                        // Skip water internal faces
                        if (isWater && neighborType === BlockType.WATER) continue;

                        // Don't render glass faces between glass blocks
                        if (blockType === BlockType.GLASS && neighborType === BlockType.GLASS) continue;

                        // Get UVs for this face
                        const faceName = faceNames[f];
                        const uv = getBlockUVs(blockType, faceName);

                        // Calculate ambient occlusion
                        const ao = this.calculateAO(wx, y, wz, face.dir);

                        // Add face vertices
                        const corners = face.corners;
                        const indices = [0, 1, 2, 0, 2, 3];

                        for (const i of indices) {
                            const corner = corners[i];
                            targetVerts.push(wx + corner[0], y + corner[1], wz + corner[2]);

                            // UV coordinates — per-face mapping
                            const uvCorner = faceUvs[f][i];
                            targetUvs.push(
                                uv.u0 + uvCorner[0] * (uv.u1 - uv.u0),
                                uv.v0 + uvCorner[1] * (uv.v1 - uv.v0)
                            );

                            // Normal
                            targetNormals.push(...face.normal);

                            // Color with AO and face lighting
                            const faceLight = this.getFaceLight(f);
                            const lightFactor = ao * faceLight;
                            targetColors.push(lightFactor, lightFactor, lightFactor);
                        }
                    }
                }
            }
        }

        // Create solid mesh
        if (vertices.length > 0) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
            geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const material = new THREE.MeshLambertMaterial({
                map: textureAtlas,
                vertexColors: true,
                side: THREE.FrontSide,
                emissive: 0xffffff,
                emissiveMap: textureAtlas,
                emissiveIntensity: 0.05
            });

            chunk.mesh = new THREE.Mesh(geometry, material);
            this.scene.add(chunk.mesh);
        }

        // Create water mesh
        if (waterVertices.length > 0) {
            const waterGeometry = new THREE.BufferGeometry();
            waterGeometry.setAttribute('position', new THREE.Float32BufferAttribute(waterVertices, 3));
            waterGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(waterUvs, 2));
            waterGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(waterNormals, 3));
            waterGeometry.setAttribute('color', new THREE.Float32BufferAttribute(waterColors, 3));

            const waterMaterial = new THREE.MeshLambertMaterial({
                map: textureAtlas,
                vertexColors: true,
                transparent: true,
                opacity: 0.7,
                side: THREE.DoubleSide,
                emissive: 0xffffff,
                emissiveMap: textureAtlas,
                emissiveIntensity: 0.05
            });

            chunk.waterMesh = new THREE.Mesh(waterGeometry, waterMaterial);
            this.scene.add(chunk.waterMesh);
        }

        // Create sprite mesh (torches, flowers, etc.)
        if (spriteVertices.length > 0) {
            const spriteGeometry = new THREE.BufferGeometry();
            spriteGeometry.setAttribute('position', new THREE.Float32BufferAttribute(spriteVertices, 3));
            spriteGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(spriteUvs, 2));
            spriteGeometry.setAttribute('normal', new THREE.Float32BufferAttribute(spriteNormals, 3));
            spriteGeometry.setAttribute('color', new THREE.Float32BufferAttribute(spriteColors, 3));

            const spriteMaterial = new THREE.MeshBasicMaterial({
                map: textureAtlas,
                vertexColors: true,
                side: THREE.DoubleSide,
                transparent: true,
                alphaTest: 0.1
            });

            chunk.spriteMesh = new THREE.Mesh(spriteGeometry, spriteMaterial);
            this.scene.add(chunk.spriteMesh);
        }

        chunk.needsUpdate = false;
    }

    // Get face light multiplier
    getFaceLight(faceIndex) {
        // Different lighting for different faces
        const lights = [0.8, 0.6, 1.0, 0.5, 0.9, 0.7];
        return lights[faceIndex];
    }

    // Calculate ambient occlusion
    calculateAO(wx, wy, wz, dir) {
        let ao = 1.0;
        let blocked = 0;

        // Check surrounding blocks
        const checks = [
            [dir[0], dir[1], dir[2] + 1],
            [dir[0], dir[1], dir[2] - 1],
            [dir[0] + 1, dir[1], dir[2]],
            [dir[0] - 1, dir[1], dir[2]],
            [dir[0], dir[1] + 1, dir[2]],
            [dir[0], dir[1] - 1, dir[2]]
        ];

        for (const [dx, dy, dz] of checks) {
            const neighbor = this.getBlock(wx + dx, wy + dy, wz + dz);
            if (isBlockSolid(neighbor)) {
                blocked += 0.1;
            }
        }

        return Math.max(0.5, 1.0 - blocked);
    }

    // Update chunks around player
    updateChunksAroundPlayer(playerX, playerZ) {
        const pcx = Math.floor(playerX / CHUNK_SIZE);
        const pcz = Math.floor(playerZ / CHUNK_SIZE);

        // Generate/load chunks in range
        for (let dx = -RENDER_DISTANCE; dx <= RENDER_DISTANCE; dx++) {
            for (let dz = -RENDER_DISTANCE; dz <= RENDER_DISTANCE; dz++) {
                const cx = pcx + dx;
                const cz = pcz + dz;
                const chunk = this.getChunk(cx, cz);

                if (!chunk) {
                    // Generate new chunk
                    const newChunk = this.createChunk(cx, cz);
                    generateTerrain(this, cx, cz);
                    newChunk.isGenerated = true;
                }
            }
        }

        // Build/update meshes
        for (const chunk of this.chunks.values()) {
            if (chunk.needsUpdate && chunk.isGenerated) {
                this.buildChunkMesh(chunk);
            }
        }

        // Remove far chunks
        for (const [key, chunk] of this.chunks) {
            const dx = chunk.cx - pcx;
            const dz = chunk.cz - pcz;
            if (Math.abs(dx) > RENDER_DISTANCE + 2 || Math.abs(dz) > RENDER_DISTANCE + 2) {
                if (chunk.mesh) {
                    this.scene.remove(chunk.mesh);
                    chunk.mesh.geometry.dispose();
                }
                if (chunk.waterMesh) {
                    this.scene.remove(chunk.waterMesh);
                    chunk.waterMesh.geometry.dispose();
                }
                if (chunk.spriteMesh) {
                    this.scene.remove(chunk.spriteMesh);
                    chunk.spriteMesh.geometry.dispose();
                }
                this.chunks.delete(key);
            }
        }
    }

    // Get all modifications for saving
    getModifications() {
        const obj = {};
        for (const [key, value] of this.modifications) {
            obj[key] = value;
        }
        return obj;
    }
}

// Global chunk manager
let chunkManager = null;

function initChunkManager(scene) {
    chunkManager = new ChunkManager(scene);
    return chunkManager;
}
