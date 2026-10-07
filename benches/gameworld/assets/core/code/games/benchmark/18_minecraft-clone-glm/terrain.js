/**
 * Procedural Terrain Generation
 * Includes biomes, caves, trees, ores
 */

// Biome types
const Biome = {
    PLAINS: 0,
    DESERT: 1,
    SNOW: 2,
    FOREST: 3
};

// Terrain generation parameters
const TERRAIN_CONFIG = {
    seaLevel: 58,
    baseHeight: 66,
    heightScale: 20,
    caveThreshold: 0.55,
    oreChance: 0.01
};

// Biome noise threshold
function getBiome(wx, wz) {
    const temp = noise.fbm2D(wx * 0.005, wz * 0.005, 4);
    const humidity = noise.fbm2D(wx * 0.005 + 1000, wz * 0.005 + 1000, 4);

    // Temperature and humidity based biome selection
    if (temp > 0.3) {
        if (humidity < -0.2) {
            return Biome.DESERT;
        }
    } else if (temp < -0.3) {
        return Biome.SNOW;
    }

    if (humidity > 0.2) {
        return Biome.FOREST;
    }

    return Biome.PLAINS;
}

// Get terrain height at world position
function getTerrainHeight(wx, wz, biome) {
    // Base terrain with multiple octaves
    let height = noise.fbm2D(wx * 0.01, wz * 0.01, 6, 2.0, 0.5);

    // Add gentle hills
    height += noise.fbm2D(wx * 0.03, wz * 0.03, 3, 2.0, 0.3) * 0.3;

    // Biome-specific modifications
    switch (biome) {
        case Biome.DESERT:
            height *= 0.5; // Flatter
            break;
        case Biome.SNOW:
            height += noise.ridged2D(wx * 0.02, wz * 0.02, 4) * 0.3; // Mountains
            break;
        case Biome.FOREST:
            height *= 0.8;
            break;
    }

    // Spawn area boost: within ~80 blocks of origin, raise terrain above sea level
    const distFromSpawn = Math.sqrt(wx * wx + wz * wz);
    if (distFromSpawn < 80) {
        const boost = (1 - distFromSpawn / 80) * 0.3;
        height = Math.max(height, boost);
    }

    // Scale and offset
    return Math.floor(TERRAIN_CONFIG.baseHeight + height * TERRAIN_CONFIG.heightScale);
}

// Check if position should be a cave
function isCave(wx, wy, wz) {
    if (wy < 5 || wy > 50) return false;

    // 3D noise for caves
    const caveNoise = noise.fbm3D(wx * 0.05, wy * 0.05, wz * 0.05, 2, 2.0, 0.5);

    // Worm-like caves
    const wormNoise = noise.noise3D(wx * 0.1, wy * 0.1, wz * 0.1);

    return Math.abs(caveNoise) > TERRAIN_CONFIG.caveThreshold && Math.abs(wormNoise) < 0.3;
}

// Get ore type at position
function getOreType(wx, wy, wz) {
    const rand = seededRandom(wx * 31 + wy * 37 + wz * 41);

    // Coal: 3-50
    if (wy >= 3 && wy <= 70 && rand() < 0.20) {
        return BlockType.COAL_ORE;
    }
    // Iron: 3-50
    if (wy >= 3 && wy <= 70 && rand() < 0.16) {
        return BlockType.IRON_ORE;
    }
    // Gold: 3-30
    if (wy >= 3 && wy <= 50 && rand() < 0.06) {
        return BlockType.GOLD_ORE;
    }
    // Diamond: 3-15
    if (wy >= 3 && wy <= 30 && rand() < 0.02) {
        return BlockType.DIAMOND_ORE;
    }

    return null;
}

// Simple seeded random
function seededRandom(seed) {
    return function() {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
    };
}

// Generate tree at position
function generateTree(chunkManager, wx, wy, wz, biome) {
    const rand = seededRandom(wx * 17 + wy * 23 + wz * 31);

    const treeHeight = 4 + Math.floor(rand() * 3);
    const logType = biome === Biome.SNOW ? BlockType.WOOD_LOG : BlockType.WOOD_LOG;

    // Check space
    for (let y = wy; y < wy + treeHeight + 2; y++) {
        if (chunkManager.getBlock(wx, y, wz) !== BlockType.AIR) return;
    }

    // Trunk
    for (let y = 0; y < treeHeight; y++) {
        chunkManager.setBlock(wx, wy + y, wz, logType);
    }

    // Leaves
    const leafType = BlockType.LEAVES;
    const leafStart = wy + treeHeight - 2;

    for (let y = leafStart; y < wy + treeHeight + 2; y++) {
        const radius = y < wy + treeHeight ? 2 : 1;
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dz = -radius; dz <= radius; dz++) {
                if (dx === 0 && dz === 0 && y < wy + treeHeight) continue;

                // Round corners
                if (Math.abs(dx) === radius && Math.abs(dz) === radius && rand() > 0.5) continue;

                const bx = wx + dx;
                const bz = wz + dz;
                if (chunkManager.getBlock(bx, y, bz) === BlockType.AIR) {
                    chunkManager.setBlock(bx, y, bz, leafType);
                }
            }
        }
    }
}

// Generate cactus in desert
function generateCactus(chunkManager, wx, wy, wz) {
    const rand = seededRandom(wx * 17 + wy * 23 + wz * 31);
    const height = 2 + Math.floor(rand() * 2);

    for (let y = 0; y < height; y++) {
        chunkManager.setBlock(wx, wy + y, wz, BlockType.CACTUS);
    }
}

// Generate terrain for a chunk
function generateTerrain(chunkManager, cx, cz) {
    const treesToGenerate = [];

    for (let lx = 0; lx < CHUNK_SIZE; lx++) {
        for (let lz = 0; lz < CHUNK_SIZE; lz++) {
            const wx = cx * CHUNK_SIZE + lx;
            const wz = cz * CHUNK_SIZE + lz;

            const biome = getBiome(wx, wz);
            const surfaceHeight = getTerrainHeight(wx, wz, biome);
            const waterLevel = TERRAIN_CONFIG.seaLevel;

            for (let ly = 0; ly < CHUNK_HEIGHT; ly++) {
                let blockType = BlockType.AIR;

                if (ly === 0) {
                    // Bedrock at bottom
                    blockType = BlockType.BEDROCK;
                } else if (ly < surfaceHeight - 4) {
                    // Deep underground - stone with ores
                    if (isCave(wx, ly, wz)) {
                        blockType = BlockType.AIR;
                    } else {
                        blockType = BlockType.STONE;
                        const ore = getOreType(wx, ly, wz);
                        if (ore) blockType = ore;
                    }
                } else if (ly < surfaceHeight - 1) {
                    // Subsurface - dirt or sand
                    if (isCave(wx, ly, wz) && ly < surfaceHeight - 4) {
                        blockType = BlockType.AIR;
                    } else {
                        blockType = biome === Biome.DESERT ? BlockType.SAND : BlockType.DIRT;
                    }
                } else if (ly < surfaceHeight) {
                    // Surface layer
                    if (biome === Biome.DESERT) {
                        blockType = BlockType.SAND;
                    } else if (biome === Biome.SNOW) {
                        blockType = BlockType.SNOW;
                    } else {
                        blockType = BlockType.GRASS;
                    }
                } else if (ly < waterLevel) {
                    // Water
                    blockType = BlockType.WATER;
                }

                // Set block
                const chunk = chunkManager.getChunk(cx, cz);
                if (chunk && blockType !== BlockType.AIR) {
                    chunk.setBlock(lx, ly, lz, blockType);
                }
            }

            // Mark tree positions
            if (surfaceHeight > waterLevel) {
                const rand = seededRandom(wx * 13 + wz * 17 + cx * 19 + cz * 23);

                if (biome === Biome.FOREST && rand() < 0.02) {
                    treesToGenerate.push({ wx, wy: surfaceHeight, wz, biome });
                } else if (biome === Biome.PLAINS && rand() < 0.005) {
                    treesToGenerate.push({ wx, wy: surfaceHeight, wz, biome });
                } else if (biome === Biome.DESERT && rand() < 0.003) {
                    treesToGenerate.push({ wx, wy: surfaceHeight, wz, isCactus: true });
                }
            }
        }
    }

    // Generate trees after all blocks are placed
    for (const tree of treesToGenerate) {
        if (tree.isCactus) {
            generateCactus(chunkManager, tree.wx, tree.wy, tree.wz);
        } else {
            generateTree(chunkManager, tree.wx, tree.wy, tree.wz, tree.biome);
        }
    }
}

// Get spawn position
function getSpawnPosition(chunkManager) {
    const spawnX = 8;
    const spawnZ = 8;

    // Find ground level
    for (let y = CHUNK_HEIGHT - 1; y >= 0; y--) {
        const block = chunkManager.getBlock(spawnX, y, spawnZ);
        if (isBlockSolid(block)) {
            return { x: spawnX + 0.5, y: y + 2, z: spawnZ + 0.5 };
        }
    }

    return { x: spawnX + 0.5, y: 80, z: spawnZ + 0.5 };
}
