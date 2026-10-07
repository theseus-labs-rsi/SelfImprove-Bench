/**
 * Inventory and Crafting System
 */

// Inventory constants
const HOTBAR_SIZE = 9;
const INVENTORY_ROWS = 3;
const INVENTORY_COLS = 9;
const INVENTORY_SIZE = INVENTORY_ROWS * INVENTORY_COLS;
const CRAFTING_SIZE = 3;

// Crafting recipes
const RECIPES = [
    // === Basic ===
    {
        // Wood log -> Wood planks (4)
        pattern: [[BlockType.WOOD_LOG]],
        result: { type: BlockType.WOOD_PLANK, count: 4 }
    },
    {
        // Wood planks -> Sticks (4)
        pattern: [
            [BlockType.WOOD_PLANK],
            [BlockType.WOOD_PLANK]
        ],
        result: { type: BlockType.STICK, count: 4 }
    },
    {
        // Wood planks -> Crafting table
        pattern: [
            [BlockType.WOOD_PLANK, BlockType.WOOD_PLANK],
            [BlockType.WOOD_PLANK, BlockType.WOOD_PLANK]
        ],
        result: { type: BlockType.CRAFTING_TABLE, count: 1 }
    },
    {
        // Stick + Coal -> Torches (4)
        pattern: [
            [BlockType.COAL_ORE],
            [BlockType.STICK]
        ],
        result: { type: BlockType.TORCH, count: 4 }
    },
    // === Wooden Tools ===
    {
        // Wooden Pickaxe
        pattern: [
            [BlockType.WOOD_PLANK, BlockType.WOOD_PLANK, BlockType.WOOD_PLANK],
            [BlockType.AIR, BlockType.STICK, BlockType.AIR],
            [BlockType.AIR, BlockType.STICK, BlockType.AIR]
        ],
        result: { type: BlockType.WOODEN_PICKAXE, count: 1 }
    },
    {
        // Wooden Axe
        pattern: [
            [BlockType.WOOD_PLANK, BlockType.WOOD_PLANK],
            [BlockType.WOOD_PLANK, BlockType.STICK],
            [BlockType.AIR, BlockType.STICK]
        ],
        result: { type: BlockType.WOODEN_AXE, count: 1 }
    },
    {
        // Wooden Sword
        pattern: [
            [BlockType.WOOD_PLANK],
            [BlockType.WOOD_PLANK],
            [BlockType.STICK]
        ],
        result: { type: BlockType.WOODEN_SWORD, count: 1 }
    },
    {
        // Wooden Shovel
        pattern: [
            [BlockType.WOOD_PLANK],
            [BlockType.STICK],
            [BlockType.STICK]
        ],
        result: { type: BlockType.WOODEN_SHOVEL, count: 1 }
    },
    {
        // Wooden Hoe
        pattern: [
            [BlockType.WOOD_PLANK, BlockType.WOOD_PLANK],
            [BlockType.AIR, BlockType.STICK],
            [BlockType.AIR, BlockType.STICK]
        ],
        result: { type: BlockType.WOODEN_HOE, count: 1 }
    },
    // === Building ===
    {
        // Cobblestone -> Furnace (simplified)
        pattern: [
            [BlockType.COBBLESTONE, BlockType.COBBLESTONE, BlockType.COBBLESTONE],
            [BlockType.COBBLESTONE, BlockType.AIR, BlockType.COBBLESTONE],
            [BlockType.COBBLESTONE, BlockType.COBBLESTONE, BlockType.COBBLESTONE]
        ],
        result: { type: BlockType.COBBLESTONE, count: 1 }
    },
    {
        // Sand -> Glass (simplified, no furnace needed)
        pattern: [
            [BlockType.SAND, BlockType.SAND],
            [BlockType.SAND, BlockType.SAND]
        ],
        result: { type: BlockType.GLASS, count: 4 }
    }
];

// Inventory state
let inventory = {
    hotbar: new Array(HOTBAR_SIZE).fill(null),
    slots: new Array(INVENTORY_SIZE).fill(null),
    crafting: new Array(CRAFTING_SIZE * CRAFTING_SIZE).fill(null),
    craftingResult: null,
    selectedSlot: 0
};

// Selected item for dragging
let draggedItem = null;
let draggedFrom = null;

// Initialize inventory with a clean start for benchmark collection tasks.
function initInventory() {
    inventory.hotbar = new Array(HOTBAR_SIZE).fill(null);
    inventory.slots = new Array(INVENTORY_SIZE).fill(null);
    inventory.crafting = new Array(CRAFTING_SIZE * CRAFTING_SIZE).fill(null);
    inventory.craftingResult = null;
    inventory.selectedSlot = 0;

    updateHotbarDisplay();
}

// Add item to inventory
function addToInventory(blockType, count) {
    // First try to stack with existing
    for (let i = 0; i < HOTBAR_SIZE; i++) {
        const slot = inventory.hotbar[i];
        if (slot && slot.type === blockType && slot.count < 64) {
            const canAdd = Math.min(count, 64 - slot.count);
            slot.count += canAdd;
            count -= canAdd;
            if (count <= 0) {
                updateHotbarDisplay();
                return true;
            }
        }
    }

    for (let i = 0; i < INVENTORY_SIZE; i++) {
        const slot = inventory.slots[i];
        if (slot && slot.type === blockType && slot.count < 64) {
            const canAdd = Math.min(count, 64 - slot.count);
            slot.count += canAdd;
            count -= canAdd;
            if (count <= 0) {
                updateHotbarDisplay();
                return true;
            }
        }
    }

    // Find empty slot
    while (count > 0) {
        let emptySlot = findEmptySlot();
        if (emptySlot === null) return false;

        const toAdd = Math.min(count, 64);
        if (emptySlot.type === 'hotbar') {
            inventory.hotbar[emptySlot.index] = { type: blockType, count: toAdd };
        } else {
            inventory.slots[emptySlot.index] = { type: blockType, count: toAdd };
        }
        count -= toAdd;
    }

    updateHotbarDisplay();
    return true;
}

// Remove item from specific slot
function removeFromInventory(slotIndex, count, isHotbar = true) {
    const array = isHotbar ? inventory.hotbar : inventory.slots;
    const slot = array[slotIndex];

    if (!slot) return false;

    slot.count -= count;
    if (slot.count <= 0) {
        array[slotIndex] = null;
    }

    updateHotbarDisplay();
    return true;
}

// Find empty slot
function findEmptySlot() {
    // Check hotbar first
    for (let i = 0; i < HOTBAR_SIZE; i++) {
        if (!inventory.hotbar[i]) {
            return { type: 'hotbar', index: i };
        }
    }

    // Check inventory
    for (let i = 0; i < INVENTORY_SIZE; i++) {
        if (!inventory.slots[i]) {
            return { type: 'inventory', index: i };
        }
    }

    return null;
}

// Hotbar selection
function selectHotbarSlot(index) {
    if (index >= 0 && index < HOTBAR_SIZE) {
        inventory.selectedSlot = index;
        updateHotbarDisplay();
    }
}

// Scroll hotbar
function scrollHotbar(delta) {
    let newSlot = inventory.selectedSlot + delta;
    if (newSlot < 0) newSlot = HOTBAR_SIZE - 1;
    if (newSlot >= HOTBAR_SIZE) newSlot = 0;
    selectHotbarSlot(newSlot);
}

// Get selected item
function getSelectedItem() {
    return inventory.hotbar[inventory.selectedSlot];
}

// Crafting system
function checkCraftingRecipe() {
    const grid = inventory.crafting;

    // Find bounds of placed items
    let minX = CRAFTING_SIZE, maxX = -1;
    let minY = CRAFTING_SIZE, maxY = -1;

    for (let y = 0; y < CRAFTING_SIZE; y++) {
        for (let x = 0; x < CRAFTING_SIZE; x++) {
            if (grid[y * CRAFTING_SIZE + x]) {
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        }
    }

    if (maxX < 0) {
        inventory.craftingResult = null;
        return null;
    }

    // Extract pattern
    const patternWidth = maxX - minX + 1;
    const patternHeight = maxY - minY + 1;
    const pattern = [];

    for (let y = 0; y < patternHeight; y++) {
        const row = [];
        for (let x = 0; x < patternWidth; x++) {
            const slot = grid[(minY + y) * CRAFTING_SIZE + (minX + x)];
            row.push(slot ? slot.type : BlockType.AIR);
        }
        pattern.push(row);
    }

    // Check against recipes
    for (const recipe of RECIPES) {
        if (matchesRecipe(pattern, recipe.pattern)) {
            inventory.craftingResult = { ...recipe.result };
            return inventory.craftingResult;
        }
    }

    inventory.craftingResult = null;
    return null;
}

// Check if pattern matches recipe
function matchesRecipe(pattern, recipePattern) {
    if (pattern.length !== recipePattern.length) return false;

    for (let y = 0; y < pattern.length; y++) {
        if (pattern[y].length !== recipePattern[y].length) return false;
        for (let x = 0; x < pattern[y].length; x++) {
            if (pattern[y][x] !== recipePattern[y][x]) return false;
        }
    }

    return true;
}

// Craft item
function craftItem() {
    if (!inventory.craftingResult) return false;

    const result = inventory.craftingResult;

    // Add result to inventory
    if (!addToInventory(result.type, result.count)) {
        return false;
    }

    // Consume crafting ingredients
    for (let i = 0; i < inventory.crafting.length; i++) {
        const slot = inventory.crafting[i];
        if (slot) {
            slot.count--;
            if (slot.count <= 0) {
                inventory.crafting[i] = null;
            }
        }
    }

    // Recheck recipe
    checkCraftingRecipe();

    playSound('pop');
    return true;
}

// Inventory UI
function initInventoryUI() {
    const inventoryGrid = document.getElementById('inventory-grid');
    const craftingGrid = document.getElementById('crafting-grid');
    const craftingResult = document.getElementById('crafting-result');
    const inventoryHotbar = document.getElementById('inventory-hotbar-grid');

    // Create inventory slots
    inventoryGrid.innerHTML = '';
    for (let i = 0; i < INVENTORY_SIZE; i++) {
        const slot = document.createElement('div');
        slot.className = 'inv-slot';
        slot.dataset.type = 'inventory';
        slot.dataset.index = i;
        inventoryGrid.appendChild(slot);
    }

    // Create hotbar slots in inventory screen
    if (inventoryHotbar) {
        inventoryHotbar.innerHTML = '';
        for (let i = 0; i < HOTBAR_SIZE; i++) {
            const slot = document.createElement('div');
            slot.className = 'inv-slot';
            slot.dataset.type = 'hotbar';
            slot.dataset.index = i;
            inventoryHotbar.appendChild(slot);
        }
    }

    // Create crafting grid
    craftingGrid.innerHTML = '';
    for (let i = 0; i < CRAFTING_SIZE * CRAFTING_SIZE; i++) {
        const slot = document.createElement('div');
        slot.className = 'inv-slot';
        slot.dataset.type = 'crafting';
        slot.dataset.index = i;
        craftingGrid.appendChild(slot);
    }

    // Crafting result slot
    craftingResult.innerHTML = '';
    craftingResult.dataset.type = 'result';

    // Add click handlers
    addInventoryClickHandlers();

    // Track mouse position for dragged item visual
    document.getElementById('inventory-screen').addEventListener('mousemove', (e) => {
        updateDraggedItemVisual(e.clientX, e.clientY);
    });
}

// Add click handlers for inventory slots
function addInventoryClickHandlers() {
    document.querySelectorAll('.inv-slot, #crafting-result').forEach(slot => {
        slot.addEventListener('mousedown', (e) => {
            e.preventDefault();
            handleInventoryClick(e, slot);
        });
        // Prevent context menu on right-click inside inventory
        slot.addEventListener('contextmenu', (e) => e.preventDefault());
    });
}

// Handle inventory click
// Left click = pick up / place whole stack
// Right click = place single item / pick up half stack
function handleInventoryClick(e, slot) {
    const type = slot.dataset.type;
    const index = parseInt(slot.dataset.index);
    const isRightClick = (e.button === 2);

    if (type === 'result') {
        // Click on crafting result
        craftItem();
        updateInventoryDisplay();
        return;
    }

    let targetArray;
    if (type === 'hotbar') {
        targetArray = inventory.hotbar;
    } else if (type === 'inventory') {
        targetArray = inventory.slots;
    } else if (type === 'crafting') {
        targetArray = inventory.crafting;
    }

    const clickedSlot = targetArray[index];

    if (draggedItem) {
        if (isRightClick) {
            // Right click with dragged item: place ONE item
            if (!clickedSlot) {
                // Empty slot — place 1
                targetArray[index] = { type: draggedItem.type, count: 1 };
                draggedItem.count--;
                if (draggedItem.count <= 0) draggedItem = null;
            } else if (clickedSlot.type === draggedItem.type && clickedSlot.count < 64) {
                // Same type — add 1
                clickedSlot.count++;
                draggedItem.count--;
                if (draggedItem.count <= 0) draggedItem = null;
            } else {
                // Different type — swap
                targetArray[index] = draggedItem;
                draggedItem = clickedSlot;
            }
        } else {
            // Left click with dragged item: place whole stack
            if (clickedSlot) {
                if (clickedSlot.type === draggedItem.type) {
                    const canAdd = Math.min(draggedItem.count, 64 - clickedSlot.count);
                    clickedSlot.count += canAdd;
                    draggedItem.count -= canAdd;
                    if (draggedItem.count <= 0) draggedItem = null;
                } else {
                    // Swap
                    targetArray[index] = draggedItem;
                    draggedItem = clickedSlot;
                }
            } else {
                targetArray[index] = draggedItem;
                draggedItem = null;
            }
        }
    } else if (clickedSlot) {
        if (isRightClick) {
            // Right click on item: pick up half
            const halfCount = Math.ceil(clickedSlot.count / 2);
            draggedItem = { type: clickedSlot.type, count: halfCount };
            clickedSlot.count -= halfCount;
            if (clickedSlot.count <= 0) {
                targetArray[index] = null;
            }
        } else {
            // Left click on item: pick up all
            draggedItem = clickedSlot;
            targetArray[index] = null;
        }
        draggedFrom = { type, index };
    }

    if (type === 'crafting') {
        checkCraftingRecipe();
    }

    updateInventoryDisplay();
    updateHotbarDisplay();
}

// Update inventory display
function updateInventoryDisplay() {
    // Update inventory slots
    const inventorySlots = document.querySelectorAll('#inventory-grid .inv-slot');
    inventorySlots.forEach((slot, i) => {
        updateSlotDisplay(slot, inventory.slots[i]);
    });

    // Update hotbar slots in inventory screen
    const hotbarSlots = document.querySelectorAll('#inventory-hotbar-grid .inv-slot');
    hotbarSlots.forEach((slot, i) => {
        updateSlotDisplay(slot, inventory.hotbar[i]);
    });

    // Update crafting slots
    const craftingSlots = document.querySelectorAll('#crafting-grid .inv-slot');
    craftingSlots.forEach((slot, i) => {
        updateSlotDisplay(slot, inventory.crafting[i]);
    });

    // Update crafting result
    const resultSlot = document.getElementById('crafting-result');
    updateSlotDisplay(resultSlot, inventory.craftingResult);

    // Update dragged item visual
    updateDraggedItemContent();
}

// Update individual slot display
function updateSlotDisplay(slotElement, slotData) {
    slotElement.innerHTML = '';

    if (slotData) {
        const icon = getBlockIcon(slotData.type);
        slotElement.appendChild(icon);

        if (slotData.count > 1) {
            const count = document.createElement('span');
            count.className = 'count';
            count.textContent = slotData.count;
            slotElement.appendChild(count);
        }
    }
}

// Update hotbar display
function updateHotbarDisplay() {
    const hotbarElement = document.getElementById('hotbar');
    hotbarElement.innerHTML = '';

    for (let i = 0; i < HOTBAR_SIZE; i++) {
        const slot = document.createElement('div');
        slot.className = 'hotbar-slot' + (i === inventory.selectedSlot ? ' selected' : '');

        const slotData = inventory.hotbar[i];
        if (slotData) {
            const icon = getBlockIcon(slotData.type);
            slot.appendChild(icon);

            if (slotData.count > 1) {
                const count = document.createElement('span');
                count.className = 'count';
                count.textContent = slotData.count;
                slot.appendChild(count);
            }
        }

        // Number key indicator
        const num = document.createElement('span');
        num.style.cssText = 'position:absolute;top:1px;left:3px;font-size:12px;font-family:VT323,monospace;color:#ffffa0;text-shadow:1px 1px 1px #000;';
        num.textContent = (i + 1) % 10;
        slot.appendChild(num);

        hotbarElement.appendChild(slot);
    }
}

// Handle inventory keyboard
function handleInventoryKey(e) {
    // Number keys 1-9
    if (e.code >= 'Digit1' && e.code <= 'Digit9') {
        const slot = parseInt(e.code.replace('Digit', '')) - 1;
        selectHotbarSlot(slot);
    }

    // E key - toggle inventory
    if (e.code === 'KeyE') {
        toggleInventory();
    }
}

// Toggle inventory screen
function toggleInventory() {
    const inventoryScreen = document.getElementById('inventory-screen');
    const isVisible = inventoryScreen.style.display === 'flex';

    if (isVisible) {
        // Close inventory
        inventoryScreen.style.display = 'none';
        uiState.isInventoryOpen = false;

        // Hide dragged item visual
        hideDraggedItemVisual();

        // Return dragged item
        if (draggedItem) {
            addToInventory(draggedItem.type, draggedItem.count);
            draggedItem = null;
        }

        // Return crafting items
        for (let i = 0; i < inventory.crafting.length; i++) {
            if (inventory.crafting[i]) {
                addToInventory(inventory.crafting[i].type, inventory.crafting[i].count);
                inventory.crafting[i] = null;
            }
        }
        inventory.craftingResult = null;

        if (game && game.isPaused) {
            game.resume();
        }
        player.requestPointerLock();
    } else {
        // Open inventory — pause FIRST to prevent onPointerLockChange from showing pause menu
        uiState.isInventoryOpen = true;
        if (game) game.pause();

        inventoryScreen.style.display = 'flex';
        updateInventoryDisplay();
        player.exitPointerLock();
    }
}

// Get inventory data for saving
function getInventoryData() {
    return {
        hotbar: inventory.hotbar,
        slots: inventory.slots
    };
}

// Load inventory from save
function loadInventoryData(data) {
    if (data.hotbar) inventory.hotbar = data.hotbar;
    if (data.slots) inventory.slots = data.slots;
    updateHotbarDisplay();
}

// Dragged item visual feedback
function updateDraggedItemVisual(mouseX, mouseY) {
    const dragEl = document.getElementById('dragged-item');
    if (!dragEl) return;

    if (draggedItem) {
        dragEl.style.display = 'block';
        dragEl.style.left = (mouseX - 16) + 'px';
        dragEl.style.top = (mouseY - 16) + 'px';
    } else {
        dragEl.style.display = 'none';
    }
}

function updateDraggedItemContent() {
    const dragEl = document.getElementById('dragged-item');
    if (!dragEl) return;

    dragEl.innerHTML = '';
    if (draggedItem) {
        dragEl.style.display = 'block';
        const icon = getBlockIcon(draggedItem.type);
        dragEl.appendChild(icon);
        if (draggedItem.count > 1) {
            const count = document.createElement('span');
            count.className = 'count';
            count.textContent = draggedItem.count;
            dragEl.appendChild(count);
        }
    } else {
        dragEl.style.display = 'none';
    }
}

function hideDraggedItemVisual() {
    const dragEl = document.getElementById('dragged-item');
    if (dragEl) {
        dragEl.style.display = 'none';
        dragEl.innerHTML = '';
    }
}
