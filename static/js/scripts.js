$(document).ready(function() {
    // Handle training form submission
    $('#training-form').submit(function(e) {
        e.preventDefault();
        
        // Show loading overlay
        $('.loading-overlay').css('display', 'flex');
        
        $.ajax({
            type: 'POST',
            url: '/train',
            data: $(this).serialize(),
            success: function(response) {
                if (response.success) {
                    // Update results
                    $('#final-reward').text(response.avg_final_reward.toFixed(2));
                    $('#learning-curve').attr('src', '/static/img/' + response.plot_url);
                    
                    // Hide loading, show results
                    $('.loading-overlay').hide();
                    $('#training-result').removeClass('d-none');
                    
                    // Scroll to results
                    $('html, body').animate({
                        scrollTop: $("#training-result").offset().top - 100
                    }, 1000);
                } else {
                    alert('Error: ' + response.message);
                    $('.loading-overlay').hide();
                }
            },
            error: function() {
                alert('Terjadi kesalahan saat melatih model. Silakan coba lagi.');
                $('.loading-overlay').hide();
            }
        });
    });
    
    // Handle parameter changes
    $('.hyperparam-slider').on('input', function() {
        var value = $(this).val();
        $(this).next('.value-display').text(value);
    });
    
    // Handle demo step navigation
    $('.step-btn').click(function() {
        var stepIndex = $(this).data('step');
        showStep(stepIndex);
    });
    
    // Auto-scroll to bottom of demo history
    $('.demo-history').scrollTop($('.demo-history')[0].scrollHeight);
});

function showStep(stepIndex) {
    // Hide all steps
    $('.history-item').removeClass('active');
    
    // Show selected step
    $('.history-item').eq(stepIndex).addClass('active');
    
    // Scroll to selected step
    var element = $('.history-item').eq(stepIndex);
    $('.demo-history').scrollTop(element.position().top);
}

// Preload images to improve performance
function preloadImages() {
    var images = [
        '/static/img/example_policy.png',
        '/static/img/example_learning_curve.png'
    ];
    
    images.forEach(function(src) {
        var img = new Image();
        img.src = src;
    });
}

// Initialize
$(window).on('load', function() {
    preloadImages();
});