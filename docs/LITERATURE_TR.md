# Literatür ve Benzer Çalışmalar

> Aktif teknik özet: [SCIENTIFIC_METHOD_TR.md](SCIENTIFIC_METHOD_TR.md), [README.md](../README.md).

Bu proje, aşağıdaki çalışma çizgisini hedefler: **etkileşim tahmini + dikkat/alt-yapı vurgulu görsel çıktı**.

## 1) Doğrudan Referanslar

1. **PepBind (JCIM 2018)**
- Başlık: *Improving sequence-based prediction of protein-peptide binding residues by introducing intrinsic disorder and a consensus method*
- DOI: https://doi.org/10.1021/acs.jcim.8b00019
- Bu projedeki etkisi: sequence tabanlı etkileşim tahmin fikri

2. **GEPPRI (repo + makale)**
- Kod: https://github.com/shima403shafiee9513/GEPPRI.method-at-Bioinformatics
- Makale PDF: `/Users/atakanemre/Downloads/makale1.pdf`
- Bu projedeki etkisi: gerçek veri formatı ve residue etiket yaklaşımı

3. **GAINET (görsel ve GAT-attention fikri)**
- Makale PDF: `/Users/atakanemre/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/BC618E1B-93D9-41D8-89AA-69238B0CCE0E/GAINET_MAkale.pdf`
- Bu projedeki etkisi: panel görsel dili (satır bazlı molekül çifti, highlight, ok, prediction metni)

## 2) Benzer / Tamamlayıcı Kaynaklar

4. **Ge et al. (Cell Reports Physical Science, 2024)**
- DOI: https://doi.org/10.1016/j.xcrp.2024.101980
- Katkı: protein-peptide interaction için structure generation + DL çerçevesi

5. **Le et al. (J Mol Graph Model, 2024)**
- DOI: https://doi.org/10.1016/j.jmgm.2024.108777
- Katkı: ProtTrans + CNN tabanlı peptide interaction site tahmini

6. **Yin et al. (RSC Chem Biol, 2024) derleme**
- URL: https://pubs.rsc.org/en/content/articlelanding/2024/cb/d3cb00256a
- Katkı: peptide-protein ML modellerinin karşılaştırmalı görünümü

## 3) Kullanılan Teknik Dokümantasyon

- RDKit API: https://www.rdkit.org/docs/api-docs.html
- PyTorch Geometric GATConv: https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.GATConv.html
- ESM GitHub: https://github.com/facebookresearch/esm
