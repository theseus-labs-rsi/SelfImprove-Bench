(function() {
    var root = typeof window !== 'undefined' ? window : this;

    if (root.JumpGame) {
        return;
    }

    root.JumpGame = {
        isRecordOpen: true,
        adRemainTime: 0,
        isAdShow: false,
        options: null,
        showInterstitial: function(options) {
            if (!options) {
                return;
            }
            if (typeof options.beforeShowAd === 'function') {
                options.beforeShowAd();
            }
            if (typeof options.afterShowAd === 'function') {
                options.afterShowAd();
            }
        },
        showReward: function(options) {
            if (!options) {
                return;
            }
            if (typeof options.beforeShowAd === 'function') {
                options.beforeShowAd();
            }
            if (typeof options.afterShowAd === 'function') {
                options.afterShowAd('rewarded');
            }
        },
        navigate: function() {},
        startup: function() {}
    };
})();
