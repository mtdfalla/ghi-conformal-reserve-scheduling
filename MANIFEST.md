# Manifest — every table and figure in the article, and what produced it

Each row names a numbered exhibit in the article, the script in this repository that
generates it, the result file it is read from, and that file's SHA-256 as shipped.
Recompute a checksum with `sha256sum <file>`; if it does not match, the file in your
checkout is not the file the article was written from.

Exhibit numbers are taken from the compiled article, not from prose. Result files are
copied when this manifest is built and never regenerated, so no number can move between
the article and this table.

**Most figures are cropped for the article.** Twelve of the fourteen numbered figures
are included as a `_notitle` crop of the file below: the title band above the axes is
removed so the caption is not repeated inside the plot. A crop has the same pixel width
as its original and is shorter; it changes no data. **Figures 1 and 14 are used
uncropped**, because the scripts that write them draw no title band. What ships here is
in every case the file the named script writes.

| Exhibit | What it is | Generating script | Result file | SHA-256 |
|---|---|---|---|---|
| **Table 1** | Structured gap analysis of the recent literature | assembled by hand from the cited sources | — | — |
| **Table 2** | Dataset summary and weather-regime composition | `code/r1/r1_s9_regimes.py` | `results/tables/r1_s9_regime_distribution.csv` | `348c478a6770a2bef31e8956138f8a0359f1e381ce32ed29e6afac960397a9c4` |
| **Table 3** | Point-forecast error on test 2024 | `code/r1/r1_p2_table2_build.py` | `results/tables/r1_p2_table2.csv` | `d1a9f83297f4bbc74f1eca5385ea97014dd489efd5976a8ae85c0af0aa055c83` |
| **Table 4** | Boosted model against the compact deep baselines, paired tests | `code/r1/r1_p2_deep_vs_gbm.py` | `results/tables/r1_p2_deep_vs_gbm.csv` | `7740e4aa6fb1f9b392b93c2edd9065739ba2035a14b7ceecdff37b9be7cf42f4` |
| **Table 5** | Per-regime coverage and sharpness at 90 % nominal | `code/r1/r1_j2_aggregate.py` | `results/tables/r1_j2_interval_metrics.csv` | `af1b429109ff4b8984cac461544841ab4c8160cb3e7bb1240b3382013462f174` |
| **Table 6** | Adaptive conformal coverage, anticipative against delayed feedback | `code/r1/r1_j2_aggregate.py` | `results/tables/r1_j2_aci_delayed.csv` | `86f4b65b66a3ee81cda424e560fb40b7be6b5fa577326d1c128c074e66cb7b90` |
| **Table 7** | External site, mirror of Table 8 | `code/r1/r1_s9_dkasc_mirror.py` | `results/tables/r1_s9_dkasc_mirror.csv` | `d368eeba8546a4cac86876faf366207c80ff7ef6b98b0c511746d7f9bc0993cf` |
| **Table 8** | Reserve-scheduling decision value at the main site | `code/r1/r1_j5_aggregate.py` | `results/tables/r1_j5_before_after.csv` | `d7e68caef1c029f0e2103c7d8257d647ad6ce12c05dcdede39ea5ac8aabc7f6b` |
| **Table 9** | Expected-cost value captured under three reserve-level protocols | `code/r1/r1_j5_aggregate.py` | `results/tables/r1_j5_protocols.csv` | `ec4b32c63ebec346fc935312e4d32038bb40483e0aae0093994bae2af8459af7` |
| **Table 10** | Calibration-set-size ablation | `code/r1/r1_j6_aggregate.py` | `results/tables/r1_j6_calib_size.csv` | `519f3e054583a080f40d943071abd9da00e0e73b537abf6c21e78eb61b83b3e4` |
| **Table 11** | Feature ablation | `code/r1/r1_j6_aggregate.py` | `results/tables/r1_j6_feature_ablation.csv` | `d938c162ee5b0c9aee34af0fb0175bce03e072f38d08d1f9b908c7375c35527d` |
| **Table 12** | Effect of capping the upper interval bound (computed per horizon by `code/r1/r1_j2_delayed.py`) | `code/r1/r1_j2_aggregate.py` | `results/tables/r1_j2_bound_cap.csv` | `3ca2cb7085b34900a42b7455477823bed896cb2f8324dac084d1169f56c4550e` |
| **Figure 1** | Weather-regime distribution, overall and by month | `code/r1/r1_s9_regimes.py` | `results/figures/r1_s9_regime_distribution.png` | `20af6404deb188b5e95917101aa305df39191e915d6bb0fe8689b5aded2ad1a7` |
| **Figure 2** | The GHI-to-PV mapping and its fit | `code/r1/r1_build_ghi_pv_map.py` | `results/figures/r1_ghi_pv_map.png` | `23487abb218e11ddf726f3e480bf54b35dedfc7b169023db419f015d2104a455` |
| **Figure 3** | Point-forecast RMSE by horizon | `code/r1/r1_p2_table2_build.py` | `results/figures/r1_p2_all_models_rmse.png` | `4e5ac8e2a4faffee4c2a1c5a2d423fc32ff15094dd6654b2417647377344d355` |
| **Figure 4** | Coverage by weather regime, all seven methods | `code/r1/r1_j2_figures.py` | `results/figures/r1_j2_picp_by_regime_5min.png` | `7fad5a06cbe872722b03c936f91a555172a38c8d34eda74ba4b80c2aab981370` |
| **Figure 5** | Monthly coverage across the test year | `code/r1/r1_j2_aggregate.py` | `results/figures/r1_j2_reliability_delayed_yulara.png` | `d12094dab0da5ac3e1fdba6f5355ca7a55abbe56693d93f8fbfb991c4f599d49` |
| **Figure 6** | Adaptive learning-rate sweep under both feedback variants | `code/r1/r1_j2_aggregate.py` | `results/figures/r1_j2_aci_gamma_delayed_yulara.png` | `6fe9dbd2b581272aa639548007811973675d877168bf2da3da45d3a533531242` |
| **Figure 7** | Example variable day with prediction intervals | **no generating script is retained — see the note below** | `results/figures/p3_example_day_bands.png` | `63725b942248a9edca4f8b65507b845256896dfdb26500af97b5500fa5742780` |
| **Figure 8** | Multivariate against univariate RMSE by horizon | `code/_j4_aggregate.py` | `results/figures/j4_rmse_by_featureset.png` | `9d2e1dd0115686c8444149f37f860c1f037595ec01a7d1796dd511355a38ed1e` |
| **Figure 9** | Per-regime calibration at both sites | `code/r1/r1_j2_figures.py` | `results/figures/r1_j3_crosssite_calibration.png` | `2683e28f997046fef2d34741b7993f60490f76ad96f19457a8bf8a4f6ada7563` |
| **Figure 10** | Risk-cost frontier at all four horizons | `code/r1/r1_j5_aggregate.py` | `results/figures/r1_j5_frontier_yulara.png` | `238a6006f81b53085da9e422965d3fe6cc0596e372bfa22a338269eecff7a13d` |
| **Figure 11** | Tail-risk value captured by method and horizon | `code/r1/r1_j5_aggregate.py` | `results/figures/r1_j5_value_captured_cvar.png` | `7bfa8dbd2b44341133fcf452f8fad717850bb34157f446b4bdd932f6e6134eb5` |
| **Figure 12** | Value captured against battery size | `code/r1/r1_j5_aggregate.py` | `results/figures/r1_j5_battery.png` | `c50e5eb41654b1df91030a1cd5313ae53501e7a373027b348bcc29487b3ad1fa` |
| **Figure 13** | Cost-ratio sweep | `code/r1/r1_j5_aggregate.py` | `results/figures/r1_j5_costratio.png` | `e8933b3aeeba6fe2586f3831a7d4cc104e3896d8a9842123a94e7a1951d8a204` |
| **Figure 14** | Coverage stability across test years | `code/r1/r1_j6_aggregate.py` | `results/figures/r1_j6_drift_picp_by_year.png` | `95f4970a96f2d0a32fbbda82499a36dde0cd449953eb9ce18cc89cda8ec1fb97` |

## One exhibit whose plotting script is not in this repository

**Figure 7** is an illustrative example-day plot from the first analysis pass. The
figure file is shipped, and the interval metrics it illustrates are in
`results/tables/p3_interval_metrics.csv` and `results/tables/p3_crps.csv`, but the
short plotting script that drew it was not retained and is not in this repository.
It is the only exhibit in the article for which that is true; the other thirteen
figures and all twelve tables map to a script above, and each mapping is checked when
this file is generated. No number in the article is affected: Figure 7 illustrates a
single day and no reported quantity is read from it.

## Every result file in this repository

The table above covers the numbered exhibits. For completeness, the SHA-256 of every
shipped result file follows, so any of them can be checked the same way.

```
2f6b9f5c14cd5c874de6e30b21f7b69527a25de1ee891ccd055fd0816dc64e14  results/tables/j2_crps.csv
3cf25d618dbcc8aa773e4bfdead81397bd55e00a3f77e3ee8e309bdc06a9992c  results/tables/j2_crps5_by_method.csv
b667edb44a7a059ec107a561785816a84fec7636d5dfc2739542f3cf89cd8bf4  results/tables/j2_interval_metrics.csv
22ae26bad9f57b1923e4dfdcaac21fd7adbb1959780bd8fd0d3064614e60788a  results/tables/j2_picp90_5min_by_regime.csv
a51df3b6724fa2056bfc9e84edc5473c8fa2bbd9529cf629802a7839ad90c86b  results/tables/j2_picp90_method_x_horizon.csv
298e4656064ebf6ee70154b5a8487f97ff165b8ce25e74990ec38a0da5416588  results/tables/j2_pinaw90_method_x_horizon.csv
2d2933679eea03ec99a2c9667302a477f5ef50089d3404a07075f7382ed5800a  results/tables/j2_reliability_over_time.csv
8912e602917a4d21c80251c6540fd21ab6cc95d110b5e19ea0c62e46d7298cea  results/tables/j3_crosssite_picp90_5min_all.csv
d1e017dc740d18b79f9a37daad9f7d54d52f4f69f7c2ef23a39105466a9f34fb  results/tables/j3_crosssite_regime_calibration.csv
3d9218e24876c5a47294ca432e01117b59174813a17fdfbd7dfe3b83317fc2db  results/tables/j3_crps.csv
e40bdaeb2bfa1e4f99142e1cb4964975d8943e9acad35975ef6a39cb331ccf05  results/tables/j3_decision_value.csv
9fa8807e840694c4ab3e154308c19e84d31a3245228b70c1473d88ed60be98aa  results/tables/j3_dkasc_picp90_5min_by_regime.csv
5ff2f50a999375a3766e61d1aa6762c8ed9d9a0b1de120082149a4a106879fa3  results/tables/j3_interval_metrics.csv
d2a2a42d9a0039263ac7e846b3a75b0fbe6c3353f3b0149d52eb892e0e986051  results/tables/j3_point_metrics.csv
ba79114f4d2708b12a612ada1f089273d7e0efac434d0be8e172cd8d4ab2aa29  results/tables/j4_crps.csv
b44a94710c3306cd5bbfcc9a18bbcc8261cedb142f4c419343235ff2f8dd03a4  results/tables/j4_dm_vs_univariate.csv
9f20c9c8d75473f4d1823ed2e5e652c1e49d9ea6032170ab6d6ed834ff7c6243  results/tables/j4_metrics_long.csv
d920a5bb3b447d3ea5b92b6b311b8c082a81151b88d0884041b83ed187caa2ec  results/tables/j4_picp90_set_x_horizon.csv
721b720630bd514919318d42c0f39c06a6418e256fb2601d139222452143caf9  results/tables/j4_pinaw90_set_x_horizon.csv
45c57cb2cab1bf72d50b3518fd982c45ca20f17b33f9f52408c92b8208e35a91  results/tables/j4_rmse_all_set_x_horizon.csv
cf86b5dc7303d437ac0855ed52d3181a4f76f15d405ec303165d393488134852  results/tables/j4_rmse_transitional_set_x_horizon.csv
34d1c71235f16fa305348898a0cdd0f47168c05acd12b3ba64467c360bcf65d7  results/tables/j5_battery_sensitivity.csv
10bae2d1e676df0ea7a8ebca958f0220dc13606f9ca0629275da98e9dfab55ee  results/tables/j5_frontier.csv
a15cc3d69e533c49d422779d98454875dc32561801a5440cbec626f19e654a44  results/tables/j5_headline_value_captured.csv
daa0e6d17153bac681a1d9c95481d0185097515ee12470d5f04061fd59681a81  results/tables/j6_calib_size.csv
a1ae1a911e1cf5f21cc621818a22e69f72a45657ff543ff1a4a3f58cc4dd8db0  results/tables/j6_drift_coverage.csv
cdcdd7b30f102794371d6e8a91bd2f29700e5f98e5dda13684f0221072234dc7  results/tables/j6_drift_summary_all.csv
0a1212907e404af423d2d3699a4e8c02329a669519608bd0c1fc8c54a4009193  results/tables/j6_drift_summary_transitional.csv
2800648800cc2380eb1ae5bbe7d67b23035f0c8d3572c6dc2e8f6470dedc84c5  results/tables/j6_feature_ablation.csv
8e8f6d630b6c4415e5ae7e69f4ed85ee2aa3649828d8adecefd3882116b3019f  results/tables/p1_cleaning_summary.csv
0ce0ee0a5382498095eec3f329a2099dd17fdb2a92a6e8209af33673bc863438  results/tables/p1_ghi_completeness_by_year.csv
c4c8f264a6ad19c6b20f0f0b6c74537662fa211c2da1d736571be30955b23c12  results/tables/p1_regime_distribution.csv
138fce12d329fe8bdec11ca1464a8e31b5d4711ebab5983178cdb1ebc4110c9b  results/tables/p1_validity_by_year.csv
f08b8091588cbb3ff73bd04e5b833cfe8d16752891033e07e33738d242bb9301  results/tables/p2_all_models_comparison.csv
aaaff17e23a103aa63ba627320feaf2d2f160006924ea139f851c17660632681  results/tables/p2_anova.csv
6482c130348f2afc22dbd12b862db994e89128b9c18c845e39ad6934e4506747  results/tables/p2_cv_perday_errors.csv
4ef95c2ccfaaa3b3dc1dcffe2002e1858198f381669dcb40be75ac157df83080  results/tables/p2_cv_summary_by_year.csv
6efcad4c264c4fc1c4d21dbb4f51722fd67d10c0e75e713f6622c5cf7a4e0d07  results/tables/p2_deep_metrics.csv
d26daf574b5c308e02a3c158cf0241d418689d64cd0e4383ccb80aa33f9bb6fc  results/tables/p2_dm_tests.csv
040a83611b1e65a5975da6155197959c9b644ecbebb2ab5e5bf4b4cee934474e  results/tables/p2_point_metrics.csv
d626829d63f931b6a2c09274a56102daf385f780e158c98b66ed5d9f9c2ce3d4  results/tables/p2_point_metrics_by_regime.csv
6daa521062faabd51380d4db789d087e9203afd4fa2dc26246bc5c49ccc08ee1  results/tables/p3_crps.csv
51fe610d7ec472bfd59235bf16e552bbed77996e4434804932cea90ab041c185  results/tables/p3_interval_metrics.csv
d8fb5e6a05d79f846bd5a24633f3f808c54bb7a34f1a038e0ed677e5a109d2a0  results/tables/p4_costofuncertainty.csv
3eb285f1be5ac3eea757a0f364f9a879b3403e94b198abb490878b13da3d1d31  results/tables/r1_j2_ace_rms_5min.csv
86f4b65b66a3ee81cda424e560fb40b7be6b5fa577326d1c128c074e66cb7b90  results/tables/r1_j2_aci_delayed.csv
ef69e354999c95c77bd1ef770bec17527fafcee4f4afb688a654d761faf36250  results/tables/r1_j2_aci_gamma.csv
3ca2cb7085b34900a42b7455477823bed896cb2f8324dac084d1169f56c4550e  results/tables/r1_j2_bound_cap.csv
757cf64aa37fa70a3dba4e561b4222efd1ff48b89c49ef57ed634090df7876db  results/tables/r1_j2_coverage_ci.csv
b37c095e80d93935d555d075e5dc8f01c5acd554ee44dd6c884b5585cef1ef31  results/tables/r1_j2_crossing.csv
478cb157b737cb30821d57c44c54118bf2ee86e272cf9b1073d057c158ae579c  results/tables/r1_j2_crossing_by_pair.csv
b52a49680b9b63d7a341fe31dbe7191d8979f789eedef12fc87b358f8c214400  results/tables/r1_j2_crps.csv
af1b429109ff4b8984cac461544841ab4c8160cb3e7bb1240b3382013462f174  results/tables/r1_j2_interval_metrics.csv
ead7b8b51dfefefe5f2213ed2f5284bdf071583a4bbea432490e33b5c220e131  results/tables/r1_j2_reliability.csv
c3c45811f0d1215231842d7d88bcf6f2c4da1d40a0af0ba56eab8421d6e8e019  results/tables/r1_j2_stats.csv
99b4eb75ae2f1cddc6d49d9318939eac9c2f4a0e0cdf8a246014e39e0e68f227  results/tables/r1_j3_dispatch.csv
51832d81cdd265d588809b652111ef13ce86dd0e7e6b7f68e91ca532f57c9615  results/tables/r1_j3_point_causal.csv
4ba1a762da1304ccc4f48e294b86bf5b1106b83b1d98b6bf0f7e4e60a256611e  results/tables/r1_j5_battery.csv
d7e68caef1c029f0e2103c7d8257d647ad6ce12c05dcdede39ea5ac8aabc7f6b  results/tables/r1_j5_before_after.csv
f4443bb019c022b721e33d0ff0b122a07f46318683922fb0796d29f3476aeeca  results/tables/r1_j5_costratio.csv
85f8779fea6174d72634996994faa8d72d9549d4cb3d773513792f2d2205fbb9  results/tables/r1_j5_cvar_ci.csv
15b6ab9f33f30a92f8737395e9607f0823f6f367c1c25726c7e45997bec03a7c  results/tables/r1_j5_daily_costs.npz
afef6f3654fbd1e468d9239c01146bff9ff7cbf3736bb9f4fce569345513cd90  results/tables/r1_j5_frontier.csv
33daa88c3853a46f571f20d2cd634575e4276f70744bf75e48e9d30709cfef77  results/tables/r1_j5_meancvar.csv
ec4b32c63ebec346fc935312e4d32038bb40483e0aae0093994bae2af8459af7  results/tables/r1_j5_protocols.csv
10b2a52712d2ede276a8a4177ba45fcf7c8f01ba63b9e501ccd22a56c979ab79  results/tables/r1_j5_r0_reproduction.csv
ade62a12dc25ea0e3697077bf8615b8a9e5c1bb32f11e5a15b53049c88ef0474  results/tables/r1_j5_theta_selection.csv
d8164216b65d454e77cacf937c8088d169e1c1fbc638dc07734a8f0addd8b764  results/tables/r1_j5_vc_diff_ci.csv
519f3e054583a080f40d943071abd9da00e0e73b537abf6c21e78eb61b83b3e4  results/tables/r1_j6_calib_size.csv
bfb9b9622df7330d28f0b06fbf904d8d10f3c24dcb9cdd7286d50694375ec36e  results/tables/r1_j6_drift_coverage.csv
bba5d80c4d90aada20690e440090753e14c76b3d3c114101a7a47937a41bcce4  results/tables/r1_j6_drift_summary_all.csv
92445f271e8d1910b365c16a31420ffa06690d69bfb322977fdb19d73b5e92f2  results/tables/r1_j6_drift_summary_transitional.csv
d938c162ee5b0c9aee34af0fb0175bce03e072f38d08d1f9b908c7375c35527d  results/tables/r1_j6_feature_ablation.csv
7383a4f96f745c79b36ca1fd2ea61939c12d82b7bbf343924d0ddb52140b2dd1  results/tables/r1_p2_anova.csv
82e7257907632bb5563dfc99bafdb94bd565c9c22fe2041a1d6da19cc180ddd6  results/tables/r1_p2_common_support.csv
935bff8da497ba007bfcaad4fd4fd26b6d4067771b4a49760bb61167e6c0632f  results/tables/r1_p2_cv_perday_errors.csv
8312643453ed94680dd89b0ac7313febea2571e458770ad9dd6e794a93f10293  results/tables/r1_p2_cv_summary_by_year.csv
1e40d51a1eedde242e4763c64afac192e51a74f2d3cdc5447e64119109e30d27  results/tables/r1_p2_deep_causal.csv
1847ac0e5b0408316786ef97d81a12e30bb8f86cfeb5e55dc8ffcb03fd3cbcdd  results/tables/r1_p2_deep_causal_full.csv
7740e4aa6fb1f9b392b93c2edd9065739ba2035a14b7ceecdff37b9be7cf42f4  results/tables/r1_p2_deep_vs_gbm.csv
5f9428075bb2d1e750d8ef585c4a7bc16f1cbc02017d2c7396b4b7b8be21afeb  results/tables/r1_p2_point_causal.csv
1e77c4b28e990cbc14c3571ab4d1213d264631485199cfa30969800636d7bfbf  results/tables/r1_p2_point_causal_by_regime.csv
d0025dbcd905cf53ae0e0f21c7746e18f75a3734678116739cf3cfe60736558f  results/tables/r1_p2_point_dm_classical.csv
d1a9f83297f4bbc74f1eca5385ea97014dd489efd5976a8ae85c0af0aa055c83  results/tables/r1_p2_table2.csv
70f8c395ccd7a27b845ab70d1d4ae692e6310e0fe262412a11ef2e05c5eddd2b  results/tables/r1_s9_anova_repeated.csv
9ad04c7f4586d418e11d46340610292d1b1ca1a704091c49deb7a80e034f2b03  results/tables/r1_s9_blockboot.csv
125fe30139c8d7cbcce173c37eb1f3cbabd6230c503121d82a57c0b352c3b349  results/tables/r1_s9_blockboot_paired.csv
4a788930845f2f5da566cf9ef63868f2d6150aea11bc7b3085099cee14551d7c  results/tables/r1_s9_causal_exposure.csv
616886729b9d9ad794d7a0644233f6851822c6e8fb1477a00f6d267c815c17e6  results/tables/r1_s9_causal_rescore.csv
d368eeba8546a4cac86876faf366207c80ff7ef6b98b0c511746d7f9bc0993cf  results/tables/r1_s9_dkasc_mirror.csv
a59354841f96269fbf49edd2bb0e7e9bab0878936c9465cf5f938d857ae15f2e  results/tables/r1_s9_multiplicity.csv
348c478a6770a2bef31e8956138f8a0359f1e381ce32ed29e6afac960397a9c4  results/tables/r1_s9_regime_distribution.csv
974592b83506f94c4bae1f724f856aa7c72b42fe77b9c163cbdde08628a1151f  results/tables/r1_s9_selection_regret.csv
dabba86059b8abd083bc98f682ae883e3930c8d62b4d4031e121b361008f496e  results/metrics/2026-06-22-clean-manifest.json
edaaccfd2ea4856159fdb63c2d218967e47536f2d00f115c2c939a8e4ec34a1e  results/metrics/2026-06-22-raw-profile.json
471ad0637f1f5708c17792cb459387603be2072ce26648dfb27b0899cbb564e3  results/metrics/2026-06-22-regime-summary.json
91f2a5857b5c885deedbe332152a819d6b724babba8e1800987efaaab6599e0a  results/metrics/j2_summary.json
2bcec6506925de250f41c0e103c8ff9d9f502ffe51b65ea12991d375b60c4f5c  results/metrics/j3_asp_clearsky_alignment.json
eb8c39c77c8620c7305e7970148197ae6760dd02f55283955840f5c1312104dc  results/metrics/j3_summary.json
79d9f8db46295c6148a713da1e12fd5b7b7f8667f88aaf796c2891ef77f08268  results/metrics/j4_summary.json
1ada53c61288a61ed84ca2190453291aefbb6b6376d64db896d85eb924433fb9  results/metrics/j5_summary.json
92b019b7d5100e967962cd9a45ddbd24b8c8ea9426a154297c201ab753a9ee64  results/metrics/j6_summary.json
d30bc9a835c50201ba7cc248e6f53a5d764c0380b7e4229bff120d8e33fa4e8a  results/metrics/p1_cleaning_summary.json
edc27641e4dd7a321e6962dcc02e9bb83f4b14241d3aec0cd878a30d3d0eaec2  results/metrics/p1_clearsky_alignment.json
d0c87acbac442a5caa3fb61ffa9328642636da393fae32feb590889e85e6d679  results/metrics/p1_eda_summary.json
1f3799bc364afe796db5e9d5fe23d5f0083a39b90eedec0b61ddbba125604d30  results/metrics/p2_cv_anova.json
0664b785916f5de511dffbdcbae45e7c92d7a0c30b232be8aa51a8e00b51ef49  results/metrics/p2_deep_metrics.json
43f1baa0bff7b89027ff8c7c441a807e89d79a342a7270beb9308ad71e0f4173  results/metrics/p2_point_metrics.json
ff866fce7219b931a5b62e6e2e6feb6c0782d8416416ba19287812f1c6ec1f3f  results/metrics/p4_ghi_pv_mapping.json
8a3fc92d77d316188bffeef5766cab88ddd31611ba9d111fc1d795eb405c1504  results/metrics/r1_bound_cap.json
798467ea1ae93fcf66ec6a01053813b574279f7ed394350ecee30ce730c4f6a8  results/metrics/r1_crossing.json
57413c5f22d4e1074c36ea926afd2f02983a111ca7bf9d2538b99f7c954918d6  results/metrics/r1_ghi_pv_fit.json
e1abbd89395854e65f5dcd54fea69a661cc0ed8f0191774d63ebc9f86bb3bd9b  results/metrics/r1_ghi_pv_fit_sensitivity.json
9a5c9bf8fc79a2b994ce02f75387a8f6e4033bea1ed4bf156f95675d0d0ba755  results/metrics/r1_j2_h1_identity.json
34d854f91746a0e88a61d9fa5cd81f79fdf996d2db78ff572eb6772dd084e829  results/metrics/r1_j5_summary.json
22889a11e84a17765d784ecef37a8b8787041dded59e612e996ba4c60798da66  results/metrics/r1_j6_summary.json
accb050ecafebcb0ee9f322f444f9804e7a126ecfa5f36b49a4e9f015d68e214  results/metrics/r1_p2_cv_anova.json
32bac576e67e26ba8e57b1c58dc0028abd20a948908486e490de4175e6d94fca  results/metrics/r1_p2_deep_causal_full_provenance.json
3fa82af5765abcf92b32571ed8447cbf8e43557c3d2edfaba3518f0cc49ea9ab  results/metrics/r1_p2_deep_causal_provenance.json
192e1b2a0d35feba4a981b1c732537e99e2e1ec6d032e6345498f6697c27e534  results/metrics/r1_p2_deep_vs_gbm.json
abab8982285f71976fe008e1aef3a4ac7d93ceedbd3af109151d8c00b45360ab  results/metrics/r1_p2_point_causal.json
55e0b6821c0a1499155efd328d0b5720d5b4992480a311399a0282dead10cd31  results/metrics/r1_s10_runlength_guard.json
ed90135b0a2d1d6d61d9551aff5d8f52a76f059352913dcc7391a64a181650b8  results/figures/j2_aci_gamma_sensitivity.png
c97b24c28dc9c315cda2a199d4da57746b18697755a543cc9aaab1ab8be21e98  results/figures/j2_picp_by_regime_5min.png
d50b8dabb515adaabbbd7ec18f0c7139b974913b5d06649e9ec8f9e4628e9e55  results/figures/j2_picp_vs_horizon.png
635d28effda203d7b2acb1fcc1b3901261f3357b6032d13f48aa0b8536dd3227  results/figures/j2_reliability_over_time.png
78c130354532248dd92371ed170ee2639e9036e9a845da968708e85a99e7c0b7  results/figures/j3_crosssite_calibration.png
34875944d70ed572994dfdea27f67c0a083b6cfb126d2d4522d159e18b6b5b92  results/figures/j3_dkasc_value_captured.png
9d2e1dd0115686c8444149f37f860c1f037595ec01a7d1796dd511355a38ed1e  results/figures/j4_rmse_by_featureset.png
ee7908d9d18a9750df0720003e649845f7c899f4f85e6dbef442a23da29bdf1b  results/figures/j5_battery_sensitivity.png
93c3d28f7e7e312b86f28ff7b0c48c8938745399fa42f5e8af761791c3571f63  results/figures/j5_risk_cost_frontier_30min.png
f9006dd1bad47d02bb3408c143ce0fa90a37ea8c905cb374429c390ac858760c  results/figures/j5_risk_cost_frontier_5min.png
eab3c264dcab357aa6299edea0ed5fb5bd0afb9655c8439f50eb789301c41e70  results/figures/j5_value_captured.png
6037c5b0d685912e61ebf341d9a73934f549618ef71b1addc4269b7e60b20b58  results/figures/j6_calib_size.png
0d6ef45b954c303ee0912c9766664bd00d001d51626df2c3c2582c28b50ad18a  results/figures/j6_drift_picp_by_year.png
67f58ed2069e3d68cfd6fb33fa53d52e19efcf76a93341c96dedfc4b0e8a54c5  results/figures/p1_example_days.png
140a640899f539338aee696059f4ce8d2c602a28c1c252f9fede3962eb78584e  results/figures/p1_ghi_diurnal_by_season.png
cea35eaff2969c1f75791f04d863b6ecd64e83151849a2a50ecedff0b716a312  results/figures/p1_ghi_vs_pv.png
aef37a787145d574ed7d1456809938666ea9044f01e782efe38e99222b107fb3  results/figures/p1_kt_distribution.png
68f6f2d971692d423ecb582973bc10a4b5a2558af10ab40b022b769148dc531d  results/figures/p1_regime_distribution.png
7e902ae739923ca19c5a4009e4fc66d5aee020afb3e889a9c0df8e66ec6410d4  results/figures/p1_validity_heatmap.png
31afb0e7db6b50360063e945ca2852b699704a9e143271308f05b069db690efe  results/figures/p2_all_models_rmse.png
f9dafb16860463cf123830a274410414e1ac505e38877be031402799a73520e9  results/figures/p2_cv_skill_by_year.png
b46c497c4b15fe213dfb94317e88a31bf9b9baec4f5ac96bfbf0c87931032eca  results/figures/p2_rmse_by_regime_60min.png
255d14fb1f103e2fb7566e69ec1724e32094ff902c6d63c686bf65a74f4f34f8  results/figures/p2_skill_vs_horizon.png
a1492cc63467793711c7bc9bf5816b5136ec75eda68eb5971323b6c169aa5e64  results/figures/p3_crps_by_horizon.png
63725b942248a9edca4f8b65507b845256896dfdb26500af97b5500fa5742780  results/figures/p3_example_day_bands.png
e7d550e103dc1df50b50f8f831ecb3f5b2b013335bf1cbb050356c2a65ed7dfd  results/figures/p3_picp_by_regime_90.png
50274d5bd74a799faa37e45a0def98b7046601fbcc4a1fc8dd0a811d8a760b87  results/figures/p3_pinaw_by_regime_90.png
4286527e03afe87641323d9d29d4593208d14ad872faa61c9de4c2a039e2604d  results/figures/p3_reliability.png
6e62b286adaef34b473bb212c806ed24cda0a4e156632c1cc3773f02581a4f15  results/figures/p4_cost_by_policy.png
a28f4a974de6147bfabf3990cfd230b5e8315c3f42a9163b1d69b0d10a158313  results/figures/p4_value_by_regime.png
308bf8fb0b4f2ab2582d227dbe187ad8aba29061f971390ae0ffe2923171d26f  results/figures/p4_value_vs_costratio.png
23487abb218e11ddf726f3e480bf54b35dedfc7b169023db419f015d2104a455  results/figures/r1_ghi_pv_map.png
2410b59b8a28d99695ea5334316d476dba81cbbf689c95af1cbf5cc2c8100da2  results/figures/r1_j2_aci_delayed_picp_asp.png
9bbe1e97c15d4877e30d1df2afe4ae49191e9062330575a3dca62e5e5202fff6  results/figures/r1_j2_aci_delayed_picp_yulara.png
55f5be1615d1dade392348ad580c11ad3132c15a26e3006fec062405aa740aa7  results/figures/r1_j2_aci_gamma_delayed_asp.png
6fe9dbd2b581272aa639548007811973675d877168bf2da3da45d3a533531242  results/figures/r1_j2_aci_gamma_delayed_yulara.png
7fad5a06cbe872722b03c936f91a555172a38c8d34eda74ba4b80c2aab981370  results/figures/r1_j2_picp_by_regime_5min.png
759ef198acf6c672689fd7bc4d8c96ec7535621d69c9ee2afaa2a9bafc7c5c1c  results/figures/r1_j2_reliability_delayed_asp.png
d12094dab0da5ac3e1fdba6f5355ca7a55abbe56693d93f8fbfb991c4f599d49  results/figures/r1_j2_reliability_delayed_yulara.png
2683e28f997046fef2d34741b7993f60490f76ad96f19457a8bf8a4f6ada7563  results/figures/r1_j3_crosssite_calibration.png
c50e5eb41654b1df91030a1cd5313ae53501e7a373027b348bcc29487b3ad1fa  results/figures/r1_j5_battery.png
e8933b3aeeba6fe2586f3831a7d4cc104e3896d8a9842123a94e7a1951d8a204  results/figures/r1_j5_costratio.png
5b0da0e0bcf97bd01683e3eb5d96577a66776040dc0d57e1f85781de5b1dd76d  results/figures/r1_j5_frontier_asp.png
238a6006f81b53085da9e422965d3fe6cc0596e372bfa22a338269eecff7a13d  results/figures/r1_j5_frontier_yulara.png
0cfa12211b927f48727659713dc9827e3fdbbc97541be6a1c2d23c01f32cf739  results/figures/r1_j5_theta_stability.png
7bfa8dbd2b44341133fcf452f8fad717850bb34157f446b4bdd932f6e6134eb5  results/figures/r1_j5_value_captured_cvar.png
a61b1e62fe9d81a3f75caae2c9c772939cb171252057462d9785c0e039bb7414  results/figures/r1_j5_value_captured_mean.png
95f4970a96f2d0a32fbbda82499a36dde0cd449953eb9ce18cc89cda8ec1fb97  results/figures/r1_j6_drift_picp_by_year.png
4e5ac8e2a4faffee4c2a1c5a2d423fc32ff15094dd6654b2417647377344d355  results/figures/r1_p2_all_models_rmse.png
0865f9b537d6c9aa0eeb0d4ee8719434edc2e9b24882683da0182441016d5604  results/figures/r1_p2_skill_vs_horizon.png
20af6404deb188b5e95917101aa305df39191e915d6bb0fe8689b5aded2ad1a7  results/figures/r1_s9_regime_distribution.png
95980d981e796b9fd74a077f6d22852e3f1f6ffedd76a043b5dc945f90d1e03f  results/reports/2026-06-22-Phase1-Data-Report.md
f8f0172db7676f40811e461a3fde2ffe8c795949147a39d0c60416e7b1dad890  results/reports/2026-06-22-Phase2-Baselines-Note.md
c3c62f854f985850329f85d773a11f4b566668241725f0d661e4ac30c0289f1d  results/reports/2026-06-22-Phase2-Summary-Report.md
8c785fe5466d3a45dec56ae0079d882cfb925ee14f4ccf174153fea5b5110aaa  results/reports/2026-06-22-Phase3-Conformal-Report.md
f74aa399952901ef9cb8e977a28760fb7d2ed4693398f1d8c0b8368cd9719c51  results/reports/2026-06-22-Phase4-DecisionValue-Report.md
af385c45a3deec16e5dd1d381297aea308a62822db53be459d41fe640b0aa7b4  results/reports/2026-06-23-J3-ExternalSite-DKASC-Report.md
7f70ac86681fc15a58d7deb533a1151347bba2419d2e0f17a94da1aadd74f4ba  results/reports/2026-06-23-J4-Multivariate-Report.md
2ac4e83c82f81c6a0bae34a4cea4a935fee56b0efc8ed05bdb75928e71b7802c  results/reports/2026-06-24-J6-Robustness-Drift-Report.md
b13006eabff13dd41117dd46f189f545df33603dd3dfc11784884d9769b7b6bd  results/reports/2026-08-16-J2-MultiHorizon-Conformal-Report.md
7404bfdc36f84286fef40e78218bb127d0a1ac3507d7f515e8782a21d4959fa3  results/reports/2026-08-16-J5-Dispatch-SoC-CVaR-Report.md
6e8d27303d91677630d5a35676f82b32eee5e063c2fac66f20694780b3302af5  results/reports/README.md
```

## The environment

`requirements.lock.txt` pins the exact versions in which the cached model predictions
were last regenerated and checked against their recorded values — the eight cached
test RMSEs reproduced to the digit, and the two feature frames to the row.
`requirements.txt` is retained for readers who want the dependency list only; the
lockfile is what `code/r1/r1_restore_env.sh` installs, and that script now fails loudly
if the installation does not succeed instead of continuing on a partial environment.

## One provenance note about the cleaner

`code/preprocessing/p2_clean.py` carries a run-length guard so that no part of a long
data gap is ever interpolated. **The results shipped here predate that guard.** They
were produced when the cleaner filled the first six steps of a gap of any length, which
affected 4,014 of 13,925 imputed GHI cells (28.8 %), the longest affected run being
3,025 steps. The size of that difference was measured rather than assumed:
`code/r1/r1_s9_causal_rescore.py` re-scores the shipped test predictions on the subset
of rows carrying no interpolated issue-time input, and
`code/r1/r1_s10_verify_runlength_guard.py` reproduces the cell counts from the shipped
artifacts and proves the guard on constructed series. Re-running the pipeline from raw
data with the guard in place therefore yields slightly different absolute levels and
the same conclusions; the article states this.
