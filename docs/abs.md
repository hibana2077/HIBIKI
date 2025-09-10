下面是一個「簡單但有深度」且縮寫為 HIBIKI 的研究題目與完整計畫草案，聚焦於 representation visualization、interpretability、Linear Regression 與 Binary Classification。

# 題目（含縮寫）

**HIBIKI：Hyperplane-Indexed Basis for Interpretable Kernel Identification**
—以「決策超平面」為錨點，將表徵分解為**決策方向**與其**核（nullspace/kernel）**，提供可視化與可證明的線性可解釋性框架。

---

## 研究核心問題

給定二元分類的線性預測器 $f(\mathbf{x})=\mathbf{w}^\top \mathbf{x}+b$（可由**線性機率模型/最小平方法**或**邏輯斯回歸**訓練），是否能構造一個**與決策超平面對齊**的表徵視覺化與度量，使我們：

1. 清楚分離「真正決策訊號」與「對決策不敏感的冗餘變異」；
2. 在數學上證明分類決策**只**取決於沿著 $\mathbf{w}$ 的投影，並量化表徵與決策的一致度？

> 線性分類幾何、LDA/Logistic 關係與最小平方法做分類的基礎背景，見 ESL 第4章與 PRML；LDA 與 Logistic 在某些分布假設下具相近邊界。([external.dandelon.com][1], [esl.hohoweiya.xyz][2], [PennState: Statistics Online Courses][3])

---

## 研究目標

* **H-對齊視覺化**：提出以 $\mathbf{w}$ 為錨的「HIBIKI-plot」，同時呈現沿 $\mathbf{w}$ 的投影 $\alpha=\hat{\mathbf{w}}\!\cdot\!\mathbf{x}$ 與在其核空間的主方向。
* **I-可解釋指標**：定義 **HIBIKI-Score**（決策對齊度）：$\mathrm{HS}=\frac{\mathrm{Var}(\mathbb{E}[\alpha|y])}{\mathrm{Var}(\alpha)}$ 與**Subspace-Angle**（表徵-決策夾角）作為可比較的量化度量。
* **B-基線對照**：與 LDA、Logistic、Linear SVM、Perceptron、Least-Squares Classifier 的視覺化與性能做系統對照。([Scikit-learn][4], [external.dandelon.com][1], [SpringerLink][5], [維基百科][6], [MIT Press Direct][7])
* **K-核識別**：以主角角(principal angles)量化「模型使用的子空間」與「資料主變異子空間」的一致/偏離。([Scipy 文件][8], [三菱電機研究所][9])
* **I-資訊一致性**：連結 SHAP/權重解釋於線性模型下的一致性，檢驗在多重共線時的失真。([NeurIPS 会议论文集][10], [arXiv][11])

---

## 主要貢獻

1. **以決策為錨的表徵分解**：提出「決策方向 + 核空間」的視覺化流程與統計指標（HIBIKI-Score、Subspace-Angle）。
2. **可證明的可解釋性**：在一般條件下證明二元線性分類**只依賴**沿 $\mathbf{w}$ 的一維投影；核空間的擾動不改變預測（但會影響回歸殘差）。
3. **實務指南**：示範在 sklearn toy/經典資料集上，如何用極簡的工具快速產生有說服力的決策對齊視覺化，並和常見基線公平比較。([Scikit-learn][12])
4. **與既有方法銜接**：將 LDA 的監督式投影、SVM margin 幾何、SHAP 線性特性，以統一視角連結。([Scikit-learn][4], [SpringerLink][5], [維基百科][6], [NeurIPS 会议论文集][10])

---

## 與現有研究之區別

* 現有解釋法多以**事後**歸因（如 SHAP/LIME）；HIBIKI 直接以**超平面幾何**建構「內生的」可解釋分解，符合「用可解釋模型取代解釋黑箱」的倡議。([Nature][13])
* 現有視覺化（PCA/LDA/t-SNE/UMAP）聚焦全域或非線性嵌入；HIBIKI 專注**決策相關的一維訊號**與**核空間冗餘**的對照，並以**主角角**做可重現的子空間比較。([Scikit-learn][14])

---

## 創新

* **HIBIKI-plot**：同圖呈現 $\alpha$ 直方圖（分群）+ 核空間第一主成分座標散點；
* **HIBIKI-Score & Angle**：提供易比較、與任意線性基線相容的指標；
* **核空間錯覺檢測**：在多重共線下示範係數/歸因可能誤導，並提出以核空間能量與角度檢測不穩定性。([arXiv][11])

---

## 理論洞見（要點與可證明性）

1. **決策只依賴一維訊號**：
   將 $\mathbf{x}$ 分解為 $\mathbf{x}=\alpha\hat{\mathbf{w}}+\mathbf{r}$，$\hat{\mathbf{w}}=\mathbf{w}/\|\mathbf{w}\|$，$\mathbf{r}\perp \hat{\mathbf{w}}$。
   則 $f(\mathbf{x})=\|\mathbf{w}\|\alpha + b$。因此二元分類 $\mathrm{sgn}(f(\mathbf{x}))$ 只由 $\alpha$ 決定；$\mathbf{r}$ 僅影響重建/回歸殘差，不影響預測。
2. **與 LDA/Logistic 的關係**：在高斯同協方差假設下，LDA 的判別方向與 Logistic 的最優超平面常相近；HIBIKI-Score 對兩者皆適用。([PennState: Statistics Online Courses][3], [external.dandelon.com][1])
3. **子空間角**：以 $\theta=\angle(\mathrm{span}(\hat{\mathbf{w}}),\,\mathcal{U})$ 衡量決策方向與任一表徵子空間（如 PCA/LDA 子空間）的對齊度；計算可用 SciPy `subspace_angles`。([Scipy 文件][8])
4. **與 SHAP 的一致性**：線性模型下，在獨立性等假設時 SHAP 近似由權重導出；若共線嚴重，則以核空間能量與 $\theta$ 揭示歸因不穩。([NeurIPS 会议论文集][10], [arXiv][11])

---

## 數學理論推演（綱要）

* **命題 A（決策不變性）**：若 $\mathbf{x}'=\mathbf{x}+\mathbf{r}$ 且 $\mathbf{r}\perp\hat{\mathbf{w}}$，則 $f(\mathbf{x}')=f(\mathbf{x})$。
* **命題 B（最小充分一維）**：對任一線性二元分類器，其充分統計量可取為 $\alpha$；任一單調變換 $g(f(\cdot))$（如邏輯斯機率）不改變分類。
* **命題 C（對齊上界）**：若 $\theta=\angle(\hat{\mathbf{w}},\mathbf{u}_{\text{LDA}})$，則在同協方差高斯下，錯分率差距對 $\sin\theta$ 呈單調（以 PRML/ESL 的 LDA 錯誤分析為基礎可示）。([Microsoft][15], [external.dandelon.com][1])

---

## 預計資料集

* **sklearn 經典**：Breast Cancer Wisconsin（容易、二元）、`make_classification`（可控共線/訊噪比）、`make_moons`（測試線性不足）。([Scikit-learn][12])

---

## Baseline 與評估

**分類 Baselines**（皆線性）：

1. 最小平方法（Linear Probability / Least-Squares Classifier，閾值0.5）；
2. Logistic Regression；
3. Linear Discriminant Analysis（LDA）；
4. Linear SVM（hinge loss）；
5. Perceptron。([external.dandelon.com][1], [Scikit-learn][4], [SpringerLink][5], [維基百科][6], [MIT Press Direct][7])

**視覺化 Baselines**：PCA（非監督）與 LDA（監督）投影圖。([Scikit-learn][14])

**指標**：Accuracy、AUC、Brier（校準）；**HIBIKI-Score**、**Subspace-Angle**（SciPy 計算）；穩健性測試（對核空間加噪）。([Scipy 文件][8])

---

## Toy-experiment 設計（含所有 Baseline）

* **資料**：

  * `make_classification(n_features=10, n_informative=1, n_redundant=7)`：控制「一維訊號 + 多重共線冗餘」；
  * `make_moons(noise=0.3)`：檢測線性不足的極端；
  * `load_breast_cancer`：真實結構驗證。([Scikit-learn][16])
* **流程**：

  1. 拆分 train/val/test（含分層）；
  2. 以 5 個分類 Baseline 各自訓練，記錄指標；
  3. 取每個 Baseline 的 $\mathbf{w}$，產生 HIBIKI-plot（$\alpha$ 直方圖 + 核空間第一主成分散點）；
  4. 算 HIBIKI-Score、Subspace-Angle；
  5. 在核空間加 i.i.d. 高斯噪音，檢測預測不變與 Brier 變化；
  6. 與 PCA/LDA 視覺化對照（是否能與決策方向對齊）。([Scikit-learn][14])

---

## 與現有研究連結（引用基礎）

* LDA/Logistic 幾何、Least-Squares 分類：**ESL Ch.4**、**PRML**。([external.dandelon.com][1], [Microsoft][15])
* SVM 與 hinge：**Cortes & Vapnik (1995)**；hinge 定義。([SpringerLink][5], [維基百科][6])
* Perceptron 歷史與線性可分：**Rosenblatt (1958)**。([MIT Press Direct][7])
* 主角角與實作：**SciPy `subspace_angles`**、Knyazev 等。([Scipy 文件][8], [三菱電機研究所][9])
* SHAP 線性特性：**Lundberg & Lee (NeurIPS 2017)**。([NeurIPS 会议论文集][10])
* 可解釋 vs. 解釋黑箱：**Rudin (2019)**。([Nature][13])

---

## 實驗成功投稿計畫

* **定位**：方法論 + 理論保證 + 可重現套件（Python/sklearn + `scipy.linalg.subspace_angles`）；
* **目標**：可先投 **Interpretable/Responsible ML** 相關的頂會工作坊（如 NeurIPS/ICML/ICLR workshops），主打「**極簡線性模型也能有深刻視覺化與理論**」；同步 arXiv 與開源程式碼/Notebook。
* **實證亮點**：在共線/高冗餘情境，HIBIKI-Score 與角度能揭露 SHAP/係數解釋不穩定之處，並給出核空間操控的反例；在 Breast Cancer 可給出更直覺的「一維疾病風險座標」。([Scikit-learn][12], [Scipy 文件][8])

## 失敗（效果不顯著）投稿備案

* **轉為負結果/教學型短文**：整理「何時線性歸因失真（共線、非線性）」的實證畫報，投教學期刊/研討會 short/DFR track 或技術報告；保留 HIBIKI-plot 作為教學工具與套件（pip 發布）。
* **資料域轉換**：若在標準 toy 集無優勢，嘗試表徵頭（linear probe on frozen embeddings）情境展示「超平面-核」思想的普適性。

---

## 可交付物

* **開源工具**：`hibiki`（產生 $\alpha$ & 核空間、HIBIKI-plot、HS 與角度），API 基於 numpy/scipy/sklearn。
* **實驗包**：所有 baseline 的訓練腳本與可復現 seed；資料一鍵下載。

---

## 風險與對策

* **多重共線 → 解釋不穩**：以核空間能量與子空間角標註不確定性；輔以 L1/Elastic-Net 作穩健性檢查。([arXiv][11])
* **線性不足**：在 `make_moons` 上示範失效，作為方法邊界的誠實報告。
* **樣本不均**：在 Breast Cancer 做分層抽樣與校準度量（Brier）。

---

## 附：實作備忘（引用官方文件）

* **資料**：`sklearn.datasets.load_breast_cancer`、toy datasets。([Scikit-learn][12])
* **視覺化基線**：`sklearn.decomposition.PCA`、`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`。([Scikit-learn][14])
* **子空間角**：`scipy.linalg.subspace_angles(A,B)`。([Scipy 文件][8])

---

[1]: https://external.dandelon.com/download/attachments/dandelon/ids/DEAGI2BDADE03FD493005C125747800699A91.pdf?utm_source=chatgpt.com "The Elements of Statistical Learning"
[2]: https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf?utm_source=chatgpt.com "The Elements of Statistical Learning"
[3]: https://online.stat.psu.edu/stat857/node/184/?utm_source=chatgpt.com "9.2.9 - Connection between LDA and logistic regression"
[4]: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html?utm_source=chatgpt.com "LinearDiscriminantAnalysis"
[5]: https://link.springer.com/article/10.1007/BF00994018?utm_source=chatgpt.com "Support-vector networks | Machine Learning"
[6]: https://en.wikipedia.org/wiki/Hinge_loss?utm_source=chatgpt.com "Hinge loss"
[7]: https://direct.mit.edu/books/edited-volume/5431/chapter/3958515/1958-F-Rosenblatt-The-perceptron-a-probabilistic?utm_source=chatgpt.com "(1958) F. Rosenblatt, The perceptron: a probabilistic model ..."
[8]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html?utm_source=chatgpt.com "subspace_angles — SciPy v1.16.1 Manual"
[9]: https://www.merl.com/publications/docs/TR2012-058.pdf?utm_source=chatgpt.com "Principal Angles Between Subspaces and Their Tangents"
[10]: https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf?utm_source=chatgpt.com "A Unified Approach to Interpreting Model Predictions"
[11]: https://arxiv.org/html/2407.12177v1?utm_source=chatgpt.com "Are Linear Regression Models White Box and Interpretable?"
[12]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html?utm_source=chatgpt.com "load_breast_cancer"
[13]: https://www.nature.com/articles/s42256-019-0048-x?utm_source=chatgpt.com "Stop explaining black box machine learning models for ..."
[14]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?utm_source=chatgpt.com "PCA — scikit-learn 1.7.2 documentation"
[15]: https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf?utm_source=chatgpt.com "Bishop-Pattern-Recognition-and-Machine-Learning-2006. ..."
[16]: https://scikit-learn.org/stable/datasets/toy_dataset.html?utm_source=chatgpt.com "8.1. Toy datasets"
