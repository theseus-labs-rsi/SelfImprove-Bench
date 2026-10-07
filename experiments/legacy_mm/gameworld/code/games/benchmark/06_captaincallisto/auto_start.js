(function(){
    if (window.__captaincallisto_auto_start) return;
    window.__captaincallisto_auto_start = true;

    const handlers = [];
    function record(target, type, listener) {
      if (!listener) return;
      if (type !== 'keydown' && type !== 'keypress' && type !== 'keyup') return;
      handlers.push({ target, type, listener });
    }

    function wrapAdd(target) {
      if (!target || typeof target.addEventListener !== 'function') return;
      if (target.addEventListener.__autoStartWrapped) return;
      const original = target.addEventListener;
      target.addEventListener = function(type, listener, options) {
        record(target, type, listener);
        return original.call(this, type, listener, options);
      };
      target.addEventListener.__autoStartWrapped = true;
    }

    function wrapEventTarget() {
      if (!window.EventTarget || !window.EventTarget.prototype) return;
      const original = window.EventTarget.prototype.addEventListener;
      if (!original || original.__autoStartWrapped) return;
      window.EventTarget.prototype.addEventListener = function(type, listener, options) {
        record(this, type, listener);
        return original.call(this, type, listener, options);
      };
      window.EventTarget.prototype.addEventListener.__autoStartWrapped = true;
    }

    function wrapOnKey(target) {
      try {
        let stored = null;
        Object.defineProperty(target, 'onkeydown', {
          configurable: true,
          get: function() { return stored; },
          set: function(fn) { stored = fn; record(target, 'keydown', fn); }
        });
      } catch (e) {}
    }

    function makeEvent(type, key, code, keyCode) {
      const event = {
        type: type,
        key: key,
        code: code,
        keyCode: keyCode,
        which: keyCode,
        bubbles: true,
        isTrusted: true,
        preventDefault: function() {},
        stopPropagation: function() {},
        stopImmediatePropagation: function() {},
        getModifierState: function() { return false; }
      };
      try {
        if (window.KeyboardEvent && window.KeyboardEvent.prototype) {
          Object.setPrototypeOf(event, window.KeyboardEvent.prototype);
        }
      } catch (e) {}
      return event;
    }

    function collectTargets() {
      const targets = [];
      targets.push(window);
      targets.push(document);
      if (document.body) {
        targets.push(document.body);
      }
      const canvases = document.querySelectorAll ? document.querySelectorAll('canvas') : [];
      for (let i = 0; i < canvases.length; i += 1) {
        targets.push(canvases[i]);
      }
      return targets;
    }

    function dispatchKeyEvent(type, key, code, keyCode) {
      if (!window.KeyboardEvent) return;
      const init = { key: key, code: code, keyCode: keyCode, which: keyCode, bubbles: true };
      const targets = collectTargets();
      for (let i = 0; i < targets.length; i += 1) {
        const target = targets[i];
        if (!target || !target.dispatchEvent) continue;
        try {
          target.dispatchEvent(new KeyboardEvent(type, init));
        } catch (e) {}
      }
    }

    function dispatchClick() {
      if (!window.MouseEvent) return;
      const targets = collectTargets();
      for (let i = 0; i < targets.length; i += 1) {
        const target = targets[i];
        if (!target || !target.dispatchEvent || !target.getBoundingClientRect) continue;
        const rect = target.getBoundingClientRect();
        const init = {
          bubbles: true,
          clientX: rect.left + rect.width / 2,
          clientY: rect.top + rect.height / 2
        };
        try {
          target.dispatchEvent(new MouseEvent('mousedown', init));
          target.dispatchEvent(new MouseEvent('mouseup', init));
          target.dispatchEvent(new MouseEvent('click', init));
        } catch (e) {}
      }
    }

    function fireHandlers(key, code, keyCode) {
      for (let i = 0; i < handlers.length; i += 1) {
        const entry = handlers[i];
        const listener = entry.listener;
        const event = makeEvent(entry.type, key, code, keyCode);
        try {
          if (typeof listener === 'function') {
            listener.call(entry.target, event);
          } else if (listener && typeof listener.handleEvent === 'function') {
            listener.handleEvent(event);
          }
        } catch (e) {}
      }
    }

    function sendKey(key, code, keyCode) {
      const types = ['keydown', 'keypress', 'keyup'];
      for (let i = 0; i < types.length; i += 1) {
        dispatchKeyEvent(types[i], key, code, keyCode);
      }
      fireHandlers(key, code, keyCode);
    }

    function schedule() {
      let attempts = 0;
      const maxAttempts = 10;
      const interval = window.setInterval(function() {
        attempts += 1;
        sendKey('Enter', 'Enter', 13);
        sendKey(' ', 'Space', 32);
        dispatchClick();
        if (attempts >= maxAttempts) {
          window.clearInterval(interval);
        }
      }, 500);
    }

    function scheduleWithDelay() {
      window.setTimeout(schedule, 1000);
    }

    wrapEventTarget();
    wrapAdd(window);
    wrapAdd(document);
    wrapOnKey(window);
    wrapOnKey(document);

    if (document.readyState === 'complete') {
      scheduleWithDelay();
    } else {
      window.addEventListener('load', scheduleWithDelay, { once: true });
    }
  })();
