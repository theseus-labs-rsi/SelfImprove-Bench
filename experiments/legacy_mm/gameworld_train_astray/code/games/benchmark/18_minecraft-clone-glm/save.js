/**
 * Save/Load System
 * localStorage for world persistence
 */

const SAVE_KEY = 'minecraft_clone_save';
const SAVE_VERSION = 1;

// Save data structure
function createSaveData() {
    return {
        version: SAVE_VERSION,
        timestamp: Date.now(),
        player: null,
        inventory: null,
        modifications: null,
        lighting: null,
        seed: null
    };
}

// Save game to localStorage
function saveGame() {
    if (!player || !chunkManager) {
        console.warn('Cannot save: game not initialized');
        return false;
    }

    try {
        const saveData = createSaveData();

        // Player data
        saveData.player = player.getPosition();

        // Inventory data
        saveData.inventory = getInventoryData();

        // Block modifications
        saveData.modifications = chunkManager.getModifications();

        // Lighting/time
        saveData.lighting = getLightingState();

        // World seed
        saveData.seed = worldSeed;

        // Save to localStorage
        localStorage.setItem(SAVE_KEY, JSON.stringify(saveData));

        console.log('Game saved successfully');
        return true;
    } catch (e) {
        console.error('Failed to save game:', e);
        return false;
    }
}

// Load game from localStorage
function loadGame() {
    try {
        const saveString = localStorage.getItem(SAVE_KEY);
        if (!saveString) {
            showNotification('No saved world found!');
            return false;
        }

        const saveData = JSON.parse(saveString);

        // Check version
        if (saveData.version !== SAVE_VERSION) {
            showNotification('Save version incompatible!');
            return false;
        }

        // Start game with saved seed
        startGame(saveData.seed, saveData);

        return true;
    } catch (e) {
        console.error('Failed to load game:', e);
        showNotification('Failed to load world!');
        return false;
    }
}

// Apply save data to game
function applySaveData(saveData) {
    if (!saveData) return;

    // Wait for world generation
    setTimeout(() => {
        // Load player position
        if (saveData.player && player) {
            player.setPosition(saveData.player);
        }

        // Load inventory
        if (saveData.inventory) {
            loadInventoryData(saveData.inventory);
        }

        // Load modifications
        if (saveData.modifications && chunkManager) {
            chunkManager.applyModifications(saveData.modifications);
        }

        // Load lighting
        if (saveData.lighting) {
            loadLightingState(saveData.lighting);
        }

        console.log('Save data applied');
    }, 1000);
}

// Check if save exists
function hasSaveData() {
    return localStorage.getItem(SAVE_KEY) !== null;
}

// Delete save data
function deleteSaveData() {
    localStorage.removeItem(SAVE_KEY);
    console.log('Save data deleted');
}

// Auto-save interval
let autoSaveInterval = null;

// Start auto-save
function startAutoSave(intervalMs = 60000) {
    if (autoSaveInterval) {
        stopAutoSave();
    }

    autoSaveInterval = setInterval(() => {
        if (game && game.isRunning && !isUIOpen()) {
            saveGame();
            console.log('Auto-saved');
        }
    }, intervalMs);
}

// Stop auto-save
function stopAutoSave() {
    if (autoSaveInterval) {
        clearInterval(autoSaveInterval);
        autoSaveInterval = null;
    }
}

// Export save as downloadable file
function exportSave() {
    const saveString = localStorage.getItem(SAVE_KEY);
    if (!saveString) {
        showNotification('No save to export!');
        return;
    }

    const blob = new Blob([saveString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `minecraft_save_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(url);
    showNotification('Save exported!');
}

// Import save from file
function importSave(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const saveData = JSON.parse(e.target.result);
            localStorage.setItem(SAVE_KEY, JSON.stringify(saveData));
            showNotification('Save imported! Click Load World to play.');
        } catch (err) {
            showNotification('Invalid save file!');
        }
    };
    reader.readAsText(file);
}

// Quick save slot system (for multiple worlds)
const SAVE_SLOTS_KEY = 'minecraft_clone_slots';

function getSaveSlots() {
    const slots = localStorage.getItem(SAVE_SLOTS_KEY);
    return slots ? JSON.parse(slots) : {};
}

function saveToSlot(slotName) {
    const saveData = createSaveData();

    if (player) saveData.player = player.getPosition();
    if (chunkManager) saveData.modifications = chunkManager.getModifications();
    saveData.inventory = getInventoryData();
    saveData.lighting = getLightingState();
    saveData.seed = worldSeed;

    const slots = getSaveSlots();
    slots[slotName] = {
        timestamp: Date.now(),
        data: saveData
    };

    localStorage.setItem(SAVE_SLOTS_KEY, JSON.stringify(slots));
    return true;
}

function loadFromSlot(slotName) {
    const slots = getSaveSlots();
    if (!slots[slotName]) return false;

    const saveData = slots[slotName].data;
    startGame(saveData.seed, saveData);
    return true;
}

function deleteSlot(slotName) {
    const slots = getSaveSlots();
    delete slots[slotName];
    localStorage.setItem(SAVE_SLOTS_KEY, JSON.stringify(slots));
}
