/**
 * Audio System
 * Procedural sound generation using Web Audio API
 */

// Audio context
let audioContext = null;
let masterGain = null;

// Sound types
const SoundType = {
    FOOTSTEP: 'footstep',
    BREAK: 'break',
    PLACE: 'place',
    JUMP: 'jump',
    LAND: 'land',
    HURT: 'hurt',
    POP: 'pop',
    MOB_HURT: 'mobHurt',
    MOB_DEATH: 'mobDeath',
    AMBIENT: 'ambient'
};

// Initialize audio
function initAudio() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        masterGain = audioContext.createGain();
        masterGain.gain.value = 0.5;
        masterGain.connect(audioContext.destination);
    } catch (e) {
        console.warn('Web Audio API not supported');
    }
}

// Resume audio context (needed after user interaction)
function resumeAudio() {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

// Play a sound
function playSound(type, param = null) {
    if (!audioContext) return;

    resumeAudio();

    switch (type) {
        case SoundType.FOOTSTEP:
            playFootstep(param);
            break;
        case SoundType.BREAK:
            playBreakSound(param);
            break;
        case SoundType.PLACE:
            playPlaceSound(param);
            break;
        case SoundType.JUMP:
            playJumpSound();
            break;
        case SoundType.LAND:
            playLandSound();
            break;
        case SoundType.HURT:
            playHurtSound();
            break;
        case SoundType.POP:
            playPopSound();
            break;
        case SoundType.MOB_HURT:
            playMobHurtSound();
            break;
        case SoundType.MOB_DEATH:
            playMobDeathSound();
            break;
    }
}

// Generate noise buffer
function generateNoiseBuffer(duration) {
    const sampleRate = audioContext.sampleRate;
    const length = sampleRate * duration;
    const buffer = audioContext.createBuffer(1, length, sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < length; i++) {
        data[i] = Math.random() * 2 - 1;
    }

    return buffer;
}

// Footstep sound based on block type
function playFootstep(blockType) {
    const now = audioContext.currentTime;

    // Create noise
    const noiseBuffer = generateNoiseBuffer(0.1);
    const noise = audioContext.createBufferSource();
    noise.buffer = noiseBuffer;

    // Filter based on block type
    const filter = audioContext.createBiquadFilter();
    filter.type = 'lowpass';

    let frequency = 800;
    let volume = 0.15;

    switch (blockType) {
        case BlockType.GRASS:
            frequency = 600;
            volume = 0.12;
            break;
        case BlockType.DIRT:
            frequency = 500;
            volume = 0.15;
            break;
        case BlockType.STONE:
        case BlockType.COBBLESTONE:
            frequency = 1200;
            volume = 0.2;
            break;
        case BlockType.SAND:
            frequency = 400;
            volume = 0.1;
            break;
        case BlockType.WOOD_PLANK:
        case BlockType.WOOD_LOG:
            frequency = 1000;
            volume = 0.15;
            break;
        case BlockType.SNOW:
            frequency = 300;
            volume = 0.08;
            break;
        default:
            frequency = 700;
            volume = 0.12;
    }

    filter.frequency.value = frequency;

    // Envelope
    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(volume, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);

    // Connect
    noise.connect(filter);
    filter.connect(gain);
    gain.connect(masterGain);

    noise.start(now);
    noise.stop(now + 0.1);
}

// Block break sound
function playBreakSound(blockType) {
    const now = audioContext.currentTime;

    // Noise for crunch
    const noiseBuffer = generateNoiseBuffer(0.15);
    const noise = audioContext.createBufferSource();
    noise.buffer = noiseBuffer;

    // Filter
    const filter = audioContext.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 800;
    filter.Q.value = 1;

    // Pitch variation based on block
    const pitchVariation = (blockType % 10) / 10;

    // Envelope
    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(masterGain);

    noise.playbackRate.value = 0.8 + pitchVariation * 0.4;
    noise.start(now);
    noise.stop(now + 0.15);
}

// Block place sound
function playPlaceSound(blockType) {
    const now = audioContext.currentTime;

    // Thud sound
    const osc = audioContext.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(150, now);
    osc.frequency.exponentialRampToValueAtTime(50, now + 0.1);

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.3, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);

    osc.connect(gain);
    gain.connect(masterGain);

    osc.start(now);
    osc.stop(now + 0.1);

    // Add noise for texture
    const noiseBuffer = generateNoiseBuffer(0.05);
    const noise = audioContext.createBufferSource();
    noise.buffer = noiseBuffer;

    const noiseGain = audioContext.createGain();
    noiseGain.gain.setValueAtTime(0.1, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);

    const filter = audioContext.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 1000;

    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(masterGain);

    noise.start(now);
    noise.stop(now + 0.05);
}

// Jump sound
function playJumpSound() {
    const now = audioContext.currentTime;

    // Quick whoosh
    const noiseBuffer = generateNoiseBuffer(0.1);
    const noise = audioContext.createBufferSource();
    noise.buffer = noiseBuffer;

    const filter = audioContext.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(800, now);
    filter.frequency.exponentialRampToValueAtTime(400, now + 0.1);
    filter.Q.value = 2;

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.15, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(masterGain);

    noise.start(now);
    noise.stop(now + 0.1);
}

// Land sound
function playLandSound() {
    const now = audioContext.currentTime;

    // Thud
    const osc = audioContext.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(100, now);
    osc.frequency.exponentialRampToValueAtTime(40, now + 0.15);

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.35, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

    osc.connect(gain);
    gain.connect(masterGain);

    osc.start(now);
    osc.stop(now + 0.15);

    // Noise
    const noiseBuffer = generateNoiseBuffer(0.1);
    const noise = audioContext.createBufferSource();
    noise.buffer = noiseBuffer;

    const filter = audioContext.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 600;

    const noiseGain = audioContext.createGain();
    noiseGain.gain.setValueAtTime(0.15, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);

    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(masterGain);

    noise.start(now);
    noise.stop(now + 0.1);
}

// Player hurt sound
function playHurtSound() {
    const now = audioContext.currentTime;

    // Ouch sound
    const osc = audioContext.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(200, now);
    osc.frequency.exponentialRampToValueAtTime(100, now + 0.2);

    const filter = audioContext.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 1000;

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(masterGain);

    osc.start(now);
    osc.stop(now + 0.2);
}

// Pop sound (for inventory)
function playPopSound() {
    const now = audioContext.currentTime;

    const osc = audioContext.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(600, now);
    osc.frequency.exponentialRampToValueAtTime(200, now + 0.1);

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);

    osc.connect(gain);
    gain.connect(masterGain);

    osc.start(now);
    osc.stop(now + 0.1);
}

// Mob hurt sound
function playMobHurtSound() {
    const now = audioContext.currentTime;

    const osc = audioContext.createOscillator();
    osc.type = 'square';
    osc.frequency.setValueAtTime(150 + Math.random() * 50, now);
    osc.frequency.exponentialRampToValueAtTime(80, now + 0.15);

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

    osc.connect(gain);
    gain.connect(masterGain);

    osc.start(now);
    osc.stop(now + 0.15);
}

// Mob death sound
function playMobDeathSound() {
    const now = audioContext.currentTime;

    // Descending tone
    const osc = audioContext.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(300, now);
    osc.frequency.exponentialRampToValueAtTime(50, now + 0.5);

    const filter = audioContext.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(2000, now);
    filter.frequency.exponentialRampToValueAtTime(200, now + 0.5);

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(masterGain);

    osc.start(now);
    osc.stop(now + 0.5);
}

// Set master volume
function setMasterVolume(volume) {
    if (masterGain) {
        masterGain.gain.value = Math.max(0, Math.min(1, volume));
    }
}

// Get master volume
function getMasterVolume() {
    return masterGain ? masterGain.gain.value : 0.5;
}
