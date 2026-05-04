/**
 * dashboard.js — CacheSec frontend helpers
 */

(function () {
  'use strict';

  const sidebar   = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('sidebarToggle');
  const backdrop  = document.getElementById('sidebar-backdrop');

  function isMobile() {
    return window.innerWidth <= 768;
  }

  function openSidebar() {
    sidebar.classList.remove('mobile-hidden', 'collapsed');
    if (isMobile() && backdrop) backdrop.classList.add('visible');
  }

  function closeSidebar() {
    if (isMobile()) {
      sidebar.classList.add('mobile-hidden');
      if (backdrop) backdrop.classList.remove('visible');
    } else {
      sidebar.classList.add('collapsed');
    }
  }

  function sidebarOpen() {
    return isMobile()
      ? !sidebar.classList.contains('mobile-hidden')
      : !sidebar.classList.contains('collapsed');
  }

  if (toggleBtn && sidebar) {
    // Start state: hidden on mobile, visible on desktop
    if (isMobile()) {
      sidebar.classList.add('mobile-hidden');
    }

    toggleBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      sidebarOpen() ? closeSidebar() : openSidebar();
    });

    // Close on backdrop tap
    if (backdrop) {
      backdrop.addEventListener('click', closeSidebar);
    }

    // Close when a nav link is clicked on mobile (navigating away)
    sidebar.querySelectorAll('.nav-link').forEach(function (link) {
      link.addEventListener('click', function () {
        if (isMobile()) closeSidebar();
      });
    });

    // Re-evaluate on resize
    window.addEventListener('resize', function () {
      if (!isMobile()) {
        // Remove mobile classes when going to desktop
        sidebar.classList.remove('mobile-hidden');
        if (backdrop) backdrop.classList.remove('visible');
      }
    });
  }

  // -------------------------------------------------------------------------
  // Bootstrap tooltips
  // -------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
      new bootstrap.Tooltip(el);
    });
  });

  // -------------------------------------------------------------------------
  // Auto-dismiss alerts after 6 seconds
  // -------------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.alert.alert-dismissible').forEach(function (alert) {
      setTimeout(function () {
        const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
        if (bsAlert) bsAlert.close();
      }, 6000);
    });
  });

  // -------------------------------------------------------------------------
  // Topbar status auto-refresh (every 10 seconds)
  // -------------------------------------------------------------------------
  function refreshStatus() {
    const camBadge = document.getElementById('cam-badge');
    const camLabel = document.getElementById('cam-label');
    const recBadge = document.getElementById('rec-badge');
    const recLabel = document.getElementById('rec-label');

    if (!camBadge) return;

    fetch('/admin/api/status')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        // "Live" reflects whether any camera is configured for streaming.
        // "Detection" status (the loop running face recognition) is separate.
        const liveCount = d.live_camera_count || 0;
        const detCount = d.detection_camera_count || 0;
        if (liveCount > 0) {
          camBadge.className = 'badge bg-success';
          if (detCount > 0) {
            camLabel.textContent = d.night_vision ? `Live · Detection · NV` : `Live · Detection`;
          } else {
            camLabel.textContent = `Live · ${liveCount} cam${liveCount === 1 ? '' : 's'}`;
          }
        } else {
          camBadge.className = 'badge bg-secondary';
          camLabel.textContent = 'No cameras';
        }

        if (recBadge && recLabel) {
          if (d.is_recording) {
            recBadge.className = 'badge bg-danger d-none d-sm-inline-flex';
            recLabel.textContent = 'REC ' + Math.round(d.recording_duration) + 's';
          } else {
            recBadge.className = 'badge bg-secondary d-none d-sm-inline-flex';
            recLabel.textContent = 'Idle';
          }
        }
      })
      .catch(function () {
        if (camLabel) camLabel.textContent = '?';
      });
  }

  document.addEventListener('DOMContentLoaded', function () {
    refreshStatus();
    setInterval(refreshStatus, 10000);
  });

})();
