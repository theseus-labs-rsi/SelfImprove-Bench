/**
 * Block Interaction System
 * DDA ray casting, block breaking/placing
 */

// Ray casting constants
const REACH_DISTANCE = 5;
const BREAK_SPEED = 1.0;
const LEFT_CLICK_PLACE_THRESHOLD_MS = 180;
const LEFT_HOLD_MINING_SPEED_MULTIPLIER = 3.0;

// Block interaction state
let targetedBlock = null;
let targetFace = null;
let breakProgress = 0;
let breakingBlock = null;
let blockHighlight = null;

// Create block highlight mesh
let breakOverlay = null;

function createBlockHighlight() {
    const group = new THREE.Group();

    // Wire frame highlight
    const geometry = new THREE.BoxGeometry(1.005, 1.005, 1.005);
    const edges = new THREE.EdgesGeometry(geometry);
    const material = new THREE.LineBasicMaterial({
        color: 0x000000,
        linewidth: 2,
        transparent: true,
        opacity: 0.6
    });
    const wireframe = new THREE.LineSegments(edges, material);
    group.add(wireframe);

    // Break progress overlay (semi-transparent dark box)
    const overlayGeo = new THREE.BoxGeometry(1.003, 1.003, 1.003);
    const overlayMat = new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0,
        side: THREE.FrontSide,
        depthWrite: false
    });
    breakOverlay = new THREE.Mesh(overlayGeo, overlayMat);
    group.add(breakOverlay);

    blockHighlight = group;
    blockHighlight.visible = false;
    return blockHighlight;
}

// DDA Ray Casting Algorithm
function raycastBlocks(origin, direction, maxDistance) {
    // Normalize direction
    direction = direction.clone().normalize();

    // Current voxel position
    let x = Math.floor(origin.x);
    let y = Math.floor(origin.y);
    let z = Math.floor(origin.z);

    // Direction to step in
    const stepX = direction.x >= 0 ? 1 : -1;
    const stepY = direction.y >= 0 ? 1 : -1;
    const stepZ = direction.z >= 0 ? 1 : -1;

    // tMax: distance to next voxel boundary
    const tDeltaX = Math.abs(1 / direction.x);
    const tDeltaY = Math.abs(1 / direction.y);
    const tDeltaZ = Math.abs(1 / direction.z);

    let tMaxX = direction.x !== 0 ?
        ((stepX > 0 ? x + 1 : x) - origin.x) / direction.x : Infinity;
    let tMaxY = direction.y !== 0 ?
        ((stepY > 0 ? y + 1 : y) - origin.y) / direction.y : Infinity;
    let tMaxZ = direction.z !== 0 ?
        ((stepZ > 0 ? z + 1 : z) - origin.z) / direction.z : Infinity;

    let t = 0;
    let lastFace = null;

    while (t < maxDistance) {
        // Check current block
        const block = chunkManager.getBlock(x, y, z);

        if (block !== BlockType.AIR && block !== BlockType.WATER) {
            return {
                block: block,
                position: { x, y, z },
                face: lastFace,
                distance: t
            };
        }

        // Step to next voxel
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t = tMaxX;
                tMaxX += tDeltaX;
                x += stepX;
                lastFace = stepX > 0 ? 'west' : 'east';
            } else {
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                z += stepZ;
                lastFace = stepZ > 0 ? 'north' : 'south';
            }
        } else {
            if (tMaxY < tMaxZ) {
                t = tMaxY;
                tMaxY += tDeltaY;
                y += stepY;
                lastFace = stepY > 0 ? 'bottom' : 'top';
            } else {
                t = tMaxZ;
                tMaxZ += tDeltaZ;
                z += stepZ;
                lastFace = stepZ > 0 ? 'north' : 'south';
            }
        }
    }

    return null;
}

// Get face normal
function getFaceNormal(face) {
    switch (face) {
        case 'top': return { x: 0, y: 1, z: 0 };
        case 'bottom': return { x: 0, y: -1, z: 0 };
        case 'north': return { x: 0, y: 0, z: -1 };
        case 'south': return { x: 0, y: 0, z: 1 };
        case 'east': return { x: 1, y: 0, z: 0 };
        case 'west': return { x: -1, y: 0, z: 0 };
        default: return { x: 0, y: 1, z: 0 };
    }
}

// Update targeted block
function updateTargetedBlock() {
    if (!player || !player.isLocked) {
        if (blockHighlight) blockHighlight.visible = false;
        targetedBlock = null;
        return;
    }

    const origin = player.camera.position.clone();
    const direction = player.getLookDirection();

    const hit = raycastBlocks(origin, direction, REACH_DISTANCE);

    if (hit) {
        targetedBlock = hit;
        targetFace = hit.face;

        // Update highlight
        if (blockHighlight) {
            blockHighlight.position.set(
                hit.position.x + 0.5,
                hit.position.y + 0.5,
                hit.position.z + 0.5
            );
            blockHighlight.visible = true;
        }
    } else {
        targetedBlock = null;
        targetFace = null;
        if (blockHighlight) blockHighlight.visible = false;
    }
}

// Break block (left click)
function breakBlock(deltaTime) {
    if (!targetedBlock) {
        breakProgress = 0;
        breakingBlock = null;
        return false;
    }

    const pos = targetedBlock.position;
    const blockType = targetedBlock.block;
    const props = BlockProperties[blockType];

    // Check if block is breakable
    if (!props || props.hardness < 0) {
        return false;
    }

    // Check if same block
    const posKey = `${pos.x},${pos.y},${pos.z}`;
    if (!breakingBlock || breakingBlock !== posKey) {
        breakProgress = 0;
        breakingBlock = posKey;
    }

    // Check held tool for speed bonus
    let toolMultiplier = 1.0;
    const heldItem = getSelectedItem();
    if (heldItem) {
        const toolProps = BlockProperties[heldItem.type];
        if (toolProps && toolProps.toolType && toolProps.toolSpeed) {
            if (props.harvestTool && toolProps.toolType === props.harvestTool) {
                // Correct tool for this block — full speed bonus
                toolMultiplier = toolProps.toolSpeed;
            } else {
                // Wrong tool — slight bonus for having any tool
                toolMultiplier = 1.2;
            }
        }
    }

    // Increment break progress
    const hardness = props.hardness || 1;
    breakProgress += (deltaTime * BREAK_SPEED * toolMultiplier) / hardness;

    // Spawn break particles
    if (Math.random() < 0.3) {
        spawnBreakParticles(pos.x + 0.5, pos.y + 0.5, pos.z + 0.5, blockType);
    }

    // Block broken
    if (breakProgress >= 1) {
        // Remove block
        chunkManager.setBlock(pos.x, pos.y, pos.z, BlockType.AIR);

        // Add to inventory
        addToInventory(blockType, 1);

        // Sound and particles
        playSound('break', blockType);
        spawnBreakParticles(pos.x + 0.5, pos.y + 0.5, pos.z + 0.5, blockType, 20);

        breakProgress = 0;
        breakingBlock = null;

        return true;
    }

    return false;
}

// Stop breaking
function stopBreaking() {
    breakProgress = 0;
    breakingBlock = null;
}

// Place block (right click)
function placeBlock() {
    if (!targetedBlock || !targetFace) return false;

    const pos = targetedBlock.position;
    const normal = getFaceNormal(targetFace);

    // Calculate placement position
    const placeX = pos.x + normal.x;
    const placeY = pos.y + normal.y;
    const placeZ = pos.z + normal.z;

    // Check if position is valid
    if (placeY < 0 || placeY >= CHUNK_HEIGHT) return false;

    // Check if position is empty
    const existingBlock = chunkManager.getBlock(placeX, placeY, placeZ);
    if (existingBlock !== BlockType.AIR && existingBlock !== BlockType.WATER) {
        return false;
    }

    // Check if player is not in the way
    const playerAABB = player.getAABB();
    const blockAABB = {
        minX: placeX,
        maxX: placeX + 1,
        minY: placeY,
        maxY: placeY + 1,
        minZ: placeZ,
        maxZ: placeZ + 1
    };

    if (aabbIntersects(playerAABB, blockAABB)) {
        return false;
    }

    // Get selected block from hotbar
    const selectedSlot = inventory.hotbar[inventory.selectedSlot];
    if (!selectedSlot || selectedSlot.count <= 0) return false;

    const blockToPlace = selectedSlot.type;

    // Cannot place tool/item types
    const props = BlockProperties[blockToPlace];
    if (props && props.isItem) return false;

    // Place block
    chunkManager.setBlock(placeX, placeY, placeZ, blockToPlace);

    // Remove from inventory
    removeFromInventory(inventory.selectedSlot, 1);

    // Sound
    playSound('place', blockToPlace);

    return true;
}

// AABB intersection check
function aabbIntersects(a, b) {
    return (
        a.minX < b.maxX &&
        a.maxX > b.minX &&
        a.minY < b.maxY &&
        a.maxY > b.minY &&
        a.minZ < b.maxZ &&
        a.maxZ > b.minZ
    );
}

// Get break progress (0-1)
function getBreakProgress() {
    return breakProgress;
}

// Mouse event handlers for block interaction
let isLeftMouseDown = false;
let isRightMouseDown = false;
let leftMouseDownStartedAtMs = 0;
let leftMouseDidMine = false;

function initBlockInteraction() {
    // Mouse down
    document.addEventListener('mousedown', (e) => {
        if (!player || !player.isLocked) return;

        if (e.button === 0) {
            isLeftMouseDown = true;
            leftMouseDownStartedAtMs = Date.now();
            leftMouseDidMine = false;
        } else if (e.button === 2) {
            // Right click - place block
            placeBlock();
            isRightMouseDown = true;
        }
    });

    // Mouse up
    document.addEventListener('mouseup', (e) => {
        if (e.button === 0) {
            const heldMs = Date.now() - leftMouseDownStartedAtMs;
            isLeftMouseDown = false;
            if (!leftMouseDidMine && heldMs < LEFT_CLICK_PLACE_THRESHOLD_MS) {
                placeBlock();
            }
            stopBreaking();
            leftMouseDownStartedAtMs = 0;
            leftMouseDidMine = false;
        } else if (e.button === 2) {
            isRightMouseDown = false;
        }
    });

    // Prevent context menu
    document.addEventListener('contextmenu', (e) => {
        e.preventDefault();
    });
}

// Update block interaction
function updateBlockInteraction(deltaTime) {
    updateTargetedBlock();

    // Short left click places a block. Holding long enough starts mining.
    if (
        isLeftMouseDown &&
        targetedBlock &&
        Date.now() - leftMouseDownStartedAtMs >= LEFT_CLICK_PLACE_THRESHOLD_MS
    ) {
        leftMouseDidMine = true;
        breakBlock(deltaTime * LEFT_HOLD_MINING_SPEED_MULTIPLIER);
    }

    // Update break overlay visual
    if (breakOverlay) {
        if (breakProgress > 0 && breakingBlock) {
            breakOverlay.material.opacity = breakProgress * 0.6;
            breakOverlay.material.color.setHex(
                breakProgress > 0.7 ? 0xff0000 :
                breakProgress > 0.4 ? 0x333333 : 0x000000
            );
        } else {
            breakOverlay.material.opacity = 0;
        }
    }
}

// Get block highlight mesh
function getBlockHighlight() {
    return blockHighlight;
}
