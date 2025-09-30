This thesis explores the integration of continuous authentication mechanisms with zero trust architecture, using keystroke dynamics as a behavioral biometric. 
Keystroke dynamics is the analysis of unique typing patterns, provides a non-intrusive, passively monitoring solution for continuous identity verification.

This strongly supports Zero trust principles by offering:
• Continuous risk evaluation based on user behavior.
• Minimized reliance on static credentials.
• Dynamic access control even after initial login.

The study evaluated a range of traditional and deep learning models Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), Convultional Neural Network (CNN), and Long Short Term Memory (LSTM) CNN-LSTM hybrids. 
Results clearly indicated that CNN-LSTM outperformed all other models across performance evaluation metrics, such as accuracy, precision, F1-score, FAR, and FRR. CNN-LSTM achieved 98.92 % accuracy, with a very low FAR (0.006) and FRR (0.0157), proving its strength in learning both spatial and temporal patterns in typing behavior

This study integrates the trained per-user CNN-LSTM models into a real-time dash web interface, enabling interactive authentication directly through a browser. 
This study uniquely emphasizes continuous user verification as an essential feature of ZTA, where authentication must be continuous, adaptive, and context-aware. The originality of this research lies in the end-to-end implementation of user-specific CNN-LSTM models with live deployment capabilities.

Moreover, the proposed method shows strong potential for real-world security critical applications:
• In online examinations, it provides a robust mechanism to continuously verify the identity of candidates and detect impersonation attempts.
• In small-scale enterprise or high-risk settings, it aids in detecting insider threats or behavioral anomalies, enabling proactive security measures.
• Against automated bot attacks or script intrusions, keystroke dynamics provide a biometric signal that distinguishes human users from bots, offering a unique form of behavioral threat mitigation.

To enhance scalability, future work should involve training on larger datasets encompassing a broader range of users, making the system more practical for real-world deployment. Additionally, upcoming models should be designed to adapt to new users typing behaviors without requiring extensive retraining or enrollment, thereby reducing computational overhead and improving usability. Integrating keystroke dynamics with other behavioral biometrics such as mouse movement or gaze tracking could increase system robustness and reduce dependency on a single biometric trait.
