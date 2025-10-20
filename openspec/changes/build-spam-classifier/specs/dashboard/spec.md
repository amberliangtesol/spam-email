# phase4-interactive-dashboard

## Summary
Create an interactive Streamlit dashboard for real-time spam classification, performance visualization, and model analysis, with cloud deployment capabilities.

## ADDED Requirements

### Streamlit Application Core
The system MUST provide a multi-page Streamlit application for comprehensive spam classification interaction.

#### Scenario: Launch Streamlit application
When running `streamlit run app/streamlit_app.py`
Then:
- Load latest trained model and vectorizer
- Display sidebar with page navigation
- Show model version and last training date
- Present clean, professional UI with consistent theming

#### Scenario: Application configuration
Given `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
```
When launching app
Then apply all theme and server settings

### Real-time Prediction Interface
The system MUST provide an intuitive interface for single message classification.

#### Scenario: Interactive text classification
Given the "Live Prediction" page
When user enters text in the input area
Then:
- Show "Analyze" button
- On click, display prediction with:
  - Spam/Ham label with color coding
  - Confidence percentage with progress bar
  - Processing time
  - Top contributing words highlighted
- Maintain history of last 10 predictions

#### Scenario: Prediction explanation
When user clicks "Explain" on a prediction
Then show:
- Top 10 positive indicators (spam words)
- Top 10 negative indicators (ham words)
- Feature weights visualization
- Decision threshold used

### Batch Processing Interface
The system MUST support CSV file upload for batch classification.

#### Scenario: CSV upload and processing
Given the "Batch Processing" page
When user uploads a CSV file
Then:
- Validate file format (must have text column)
- Show preview of first 5 rows
- Display progress bar during processing
- Allow download of results with predictions added
- Show summary statistics (spam count, ham count)

#### Scenario: Large file handling
When uploading file with >1000 messages
Then:
- Process in chunks to avoid timeout
- Update progress incrementally
- Allow cancellation during processing
- Provide estimated completion time

### Model Performance Dashboard
The system MUST visualize comprehensive model performance metrics.

#### Scenario: Metrics overview page
Given the "Model Performance" page
When loaded
Then display:
- Key metrics cards (Accuracy, Precision, Recall, F1)
- Confusion matrix heatmap
- Classification report table
- Model training history (if available)

#### Scenario: Interactive confusion matrix
When displaying confusion matrix
Then:
- Use Plotly for interactivity
- Show counts on hover
- Include percentage annotations
- Allow export as image

### Visualization Suite
The system MUST provide multiple visualization types for analysis.

#### Scenario: PR and ROC curves
Given the "Performance Curves" section
When rendered
Then show:
- Precision-Recall curve with AUC
- ROC curve with AUC
- Optimal threshold points marked
- Interactive hover for threshold values
- Comparison with baseline (if exists)

#### Scenario: Feature importance visualization
Given the "Feature Analysis" section
When loaded
Then display:
- Top 20 most important features (bar chart)
- Word cloud of spam indicators
- Word cloud of ham indicators
- N-gram frequency comparison

#### Scenario: Class distribution charts
When showing data statistics
Then include:
- Pie chart of spam vs ham in training data
- Bar chart of message length distribution
- Histogram of prediction confidence scores

### Threshold Tuning Interface
The system MUST provide interactive threshold adjustment.

#### Scenario: Interactive threshold slider
Given the "Threshold Tuning" page
When user adjusts threshold slider (0.0 to 1.0)
Then dynamically update:
- Precision value
- Recall value
- F1 score
- Number of spam/ham predictions
- Confusion matrix preview

#### Scenario: Threshold recommendation
When viewing threshold tuning
Then show:
- Current production threshold
- Recommended threshold for max F1
- Recommended threshold for target precision
- Recommended threshold for target recall

### Data Exploration Tools
The system MUST enable exploration of the training dataset.

#### Scenario: Dataset statistics view
Given the "Data Explorer" page
When loaded
Then show:
- Total messages count
- Spam/Ham distribution
- Average message length by class
- Most common words per class
- Sample messages from each class

#### Scenario: Message search and filter
When using data explorer
Then allow:
- Search messages by keyword
- Filter by predicted class
- Filter by confidence level
- Sort by various metrics
- Export filtered results

### Model Comparison View
The system MUST compare different model versions.

#### Scenario: Multi-model comparison
Given multiple trained models exist
When selecting "Model Comparison"
Then display:
- Side-by-side metrics table
- Performance trend charts
- Training time comparison
- Model size comparison
- Best model recommendation

### Cloud Deployment Configuration
The system MUST be deployable to Streamlit Cloud.

#### Scenario: Streamlit Cloud deployment
Given deployment configuration files:
- `requirements.txt` with all dependencies
- `.streamlit/config.toml` with settings
- `app/streamlit_app.py` as entry point
When deploying to Streamlit Cloud
Then:
- App accessible at public URL
- Auto-reload on git push
- Environment variables properly set
- Model files accessible

#### Scenario: Local model serving
When running in cloud environment
Then:
- Load model from `models/` directory
- Cache model in memory
- Implement proper error handling
- Show fallback UI if model missing

### User Experience Features
The system MUST provide excellent user experience.

#### Scenario: Loading states
When performing long operations
Then show:
- Spinner with descriptive message
- Progress bars where applicable
- Estimated time remaining
- Cancel option for batch operations

#### Scenario: Error handling
When errors occur
Then:
- Display user-friendly error messages
- Provide suggested actions
- Log technical details (hidden by default)
- Maintain app stability (no crashes)

#### Scenario: Responsive design
When viewing on different devices
Then:
- Adapt layout for mobile screens
- Maintain functionality on tablets
- Optimize charts for screen size
- Ensure text remains readable

### Export and Reporting
The system MUST support various export options.

#### Scenario: Generate PDF report
When clicking "Generate Report"
Then create PDF containing:
- Model performance summary
- Key visualizations
- Configuration details
- Timestamp and version info

#### Scenario: Export visualizations
When viewing any chart
Then allow:
- Download as PNG
- Download as SVG
- Copy to clipboard
- Save plot data as CSV

## Acceptance Criteria
- [ ] Streamlit app runs locally without errors
- [ ] All pages load with proper data
- [ ] Real-time prediction working with <100ms response
- [ ] Batch processing handles 1000+ messages
- [ ] All visualizations render correctly
- [ ] Threshold tuning updates in real-time
- [ ] Successfully deployed to Streamlit Cloud
- [ ] Mobile-responsive design verified
- [ ] Export functions working for all charts
- [ ] Error handling prevents app crashes

## Dependencies
- streamlit >= 1.28.0
- plotly >= 5.17.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- wordcloud >= 1.9.0

## Deployment Files Required
```
.streamlit/
  config.toml
  secrets.toml (for API keys if needed)
requirements.txt
app/
  streamlit_app.py
  pages/
    1_Live_Prediction.py
    2_Batch_Processing.py
    3_Model_Performance.py
    4_Threshold_Tuning.py
    5_Data_Explorer.py
models/
  spam_classifier_v1.0.0.pkl
  tfidf_vectorizer_v1.0.0.pkl
  config_v1.0.0.json
```

## Related Specs
- Phase 1: Provides base model and API
- Phase 2: Provides improved recall model
- Phase 3: Provides optimized precision model
- All phases contribute models for comparison