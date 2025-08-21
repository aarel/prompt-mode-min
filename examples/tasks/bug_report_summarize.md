# Bug Report: Null Input Crash

**Summary**  
Application throws an exception when provided with `null` input.  

**Steps to Reproduce**  
1. Launch the app normally.  
2. Submit a form with no value (null input).  
3. Observe that the app crashes with a stack trace.  

**Observed Behavior**  

**Expected Behavior**  
Application should handle `null` input gracefully, e.g., return a validation error message rather than crashing.  

**Notes**  
This issue occurs in version `1.3.2`. QA reported it blocks regression testing.