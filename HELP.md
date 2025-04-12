# Deepfake Detection System - Help Guide

## Quick Reference

1. **Upload an image or video** using the upload widget in the sidebar
2. **Wait for the analysis** to complete - a progress bar will be shown
3. **View the results** - the system will show if the content is real or fake
4. **Explore the detailed analysis** in the various tabs

## FAQs

### Q: Why does the application show an error when uploading large files?
**A:** The application has a file size limit. Try to keep videos under 100MB and images under 5MB.

### Q: Why is the analysis taking so long?
**A:** Deepfake detection requires complex analysis. Videos take longer than images due to frame-by-frame processing.

### Q: How accurate is the detection?
**A:** The system achieves approximately 95% accuracy on benchmark datasets, but results may vary with different types of content.

### Q: Can I analyze any type of image or video?
**A:** The system works best with content that contains faces. It may not provide reliable results for images without faces or heavily obscured faces.

### Q: What types of deepfakes can it detect?
**A:** The system can detect:
- Face swaps
- AI-generated faces
- Manipulated facial expressions
- Digital makeup/retouching
- Cultural clothing inconsistencies (sarees, jewelry)

## Interpreting Results

### Confidence Score
- **0-30%**: Likely real
- **30-70%**: Uncertain/possibly manipulated
- **70-100%**: Likely fake

### Heatmap
Red areas in the heatmap indicate regions that the system has identified as potentially manipulated.

### Cultural Clothing Analysis
This tab shows analysis specific to cultural clothing elements like sarees and jewelry, highlighting pattern inconsistencies.

## Technical Support

If you encounter any issues:
1. Check the terminal/console window for error messages
2. Make sure your Python environment is properly set up
3. Verify that all model files were downloaded correctly

For bug reports or feature requests, please submit an issue on GitHub:
[https://github.com/ombhonsle16/deepfake](https://github.com/ombhonsle16/deepfake) 