'use strict';

document.body.classList.add('js-enabled');

const menuButton = document.querySelector('.menu-toggle');
const nav = document.querySelector('#main-nav');

if (menuButton && nav) {
  menuButton.hidden = false;

  const closeMenu = () => {
    nav.classList.remove('open');
    menuButton.setAttribute('aria-expanded', 'false');
  };

  menuButton.addEventListener('click', () => {
    const open = !nav.classList.contains('open');
    nav.classList.toggle('open', open);
    menuButton.setAttribute('aria-expanded', String(open));
  });

  nav.addEventListener('click', event => {
    if (event.target.closest('a')) closeMenu();
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && nav.classList.contains('open')) {
      closeMenu();
      menuButton.focus();
    }
  });

  const mobileQuery = matchMedia('(max-width: 800px)');
  mobileQuery.addEventListener('change', closeMenu);

  const navLinks = Array.from(nav.querySelectorAll('a[href^="#"]'));
  const sections = navLinks
    .map(link => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);
  let framePending = false;

  const updateNavigation = () => {
    let current = null;
    sections.forEach(section => {
      if (section.getBoundingClientRect().top < 180) current = section.id;
    });
    navLinks.forEach(link => {
      const active = link.getAttribute('href') === `#${current}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    framePending = false;
  };

  window.addEventListener('scroll', () => {
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(updateNavigation);
  }, { passive: true });
  updateNavigation();
}

const table = document.querySelector('.leaderboard');
const tableBody = document.querySelector('#leaderboard-body');
const methodFilter = document.querySelector('#method-filter');
const status = document.querySelector('#leaderboard-status');
const levelButtons = Array.from(document.querySelectorAll('.level-button'));
const levelHeadings = Array.from(document.querySelectorAll('.hci-heading'));

if (table && tableBody && methodFilter && status) {
  const rows = Array.from(tableBody.rows);
  let currentLevel = 'g1';
  let currentMethod = 'all';

  const levelNames = {
    g0: 'G0 (replay)',
    g1: 'G1 (new instances)',
    g2: 'G2 (harder instances)',
    g3: 'G3 (same-capability transfer)'
  };

  const numericValue = (row, level) => {
    const raw = row.dataset[level];
    return raw === '' || raw === undefined ? Number.NEGATIVE_INFINITY : Number(raw);
  };

  const renderLeaderboard = () => {
    const visible = rows.filter(row => currentMethod === 'all' || row.dataset.method === currentMethod);
    const hidden = rows.filter(row => !visible.includes(row));
    visible.sort((a, b) => numericValue(b, currentLevel) - numericValue(a, currentLevel) || numericValue(b, 'g1') - numericValue(a, 'g1'));
    visible.forEach((row, index) => {
      row.hidden = false;
      row.querySelector('.rank').textContent = String(index + 1).padStart(2, '0');
      tableBody.append(row);
    });
    hidden.forEach(row => {
      row.hidden = true;
      tableBody.append(row);
    });

    table.dataset.level = currentLevel;
    levelButtons.forEach(button => {
      const active = button.dataset.level === currentLevel;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    levelHeadings.forEach(heading => {
      const active = heading.dataset.headingLevel === currentLevel;
      heading.classList.toggle('is-active', active);
      if (active) heading.setAttribute('aria-sort', 'descending');
      else heading.setAttribute('aria-sort', 'none');
    });
    const methodName = methodFilter.options[methodFilter.selectedIndex].textContent;
    status.textContent = `Ranked by ${levelNames[currentLevel]}. ${visible.length} ${visible.length === 1 ? 'entry' : 'entries'} shown${currentMethod === 'all' ? '' : ` for ${methodName}`}.`;
  };

  levelButtons.forEach(button => {
    button.addEventListener('click', () => {
      currentLevel = button.dataset.level;
      renderLeaderboard();
    });
  });

  methodFilter.addEventListener('change', () => {
    currentMethod = methodFilter.value;
    renderLeaderboard();
  });

  renderLeaderboard();
}

const dialog = document.querySelector('#figure-dialog');
const dialogImage = document.querySelector('#dialog-image');
const dialogCaption = document.querySelector('#figure-caption');
const viewport = document.querySelector('.figure-viewport');
const zoomLabel = document.querySelector('#zoom-level');
const closeButton = document.querySelector('#close-figure');
let lightboxOpener = null;
let zoom = 1;
let fitWidth = 0;

if (dialog && dialog.showModal) {
  const fitFigure = () => {
    zoom = 1;
    dialogImage.classList.add('fit');
    dialogImage.style.width = '';
    dialogImage.style.height = '';
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
    zoomLabel.textContent = 'Fit';
    if (dialogImage.naturalWidth) {
      const availableWidth = viewport.clientWidth - 36;
      const availableHeight = viewport.clientHeight - 36;
      fitWidth = Math.min(availableWidth, availableHeight * dialogImage.naturalWidth / dialogImage.naturalHeight);
    }
  };

  const setZoom = value => {
    zoom = Math.max(1, Math.min(4, value));
    if (zoom === 1) {
      fitFigure();
      return;
    }
    if (!fitWidth) fitFigure();
    dialogImage.classList.remove('fit');
    dialogImage.style.width = `${fitWidth * zoom}px`;
    dialogImage.style.height = 'auto';
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  };

  document.querySelectorAll('[data-lightbox]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      lightboxOpener = link;
      const image = link.querySelector('img');
      dialogCaption.textContent = link.dataset.caption || '';
      dialogImage.alt = image ? image.alt : '';
      dialogImage.src = link.href;
      dialog.showModal();
      document.body.classList.add('modal-open');
      fitFigure();
      closeButton.focus();
    });
  });

  dialogImage.addEventListener('load', fitFigure);
  closeButton.addEventListener('click', () => dialog.close());
  document.querySelector('#zoom-in').addEventListener('click', () => setZoom(zoom + .5));
  document.querySelector('#zoom-out').addEventListener('click', () => setZoom(zoom - .5));
  document.querySelector('#zoom-fit').addEventListener('click', fitFigure);
  dialog.addEventListener('click', event => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('modal-open');
    if (lightboxOpener) lightboxOpener.focus();
  });
}
