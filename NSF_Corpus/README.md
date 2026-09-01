*Español* | [English](/NSF_Corpus/NSF_Corpus_README_en.md)
# <img width="80" height="97" alt="logo_NSF_english" src="/img/logo_NSF_en.png"> NSF Corpus
El Corpus está compuesto por archivos de texto (.txt) que contienen las transcripciones de los documentos históricos utilizados en el proyecto.

El **Corpus_GT** contiene las versiones "Ground Truth" de los documentos.
Estas son las transcripciones literales hechas por especialistas en paleografía novohispana, siguiendo los Criterios de Transcripción del proyecto.

Los diferentes **Corpus_HTR** contienen las transcripciones automáticas generadas por nuestros modelos de reconocimiento de texto manuscrito (HTR) alojados en [Transkribus](https://app.transkribus.eu).
El nombre de cada corpus indica el modelo HTR que generó las transcripciones. Por ejemplo, las transcripciones incluidas en "Corpus_Encadenada_m2t4" fueron producidas por el modelo HTR para letra Procesal encadenada, versión m2t4.  

La mayoría de los archivos GT y HTR existen en parejas (por ejemplo, BnF_110_02_GT -- BnF_110_02_Procesal_m3t4_HTR), pues la precisión de una transcripción HTR se evalúa comparándola con su GT.  
Sin embargo, para un mismo archivo GT pueden existir uno o más archivos HTR, en caso de que se hayan generado varias transcripciones de una misma página usando diferentes modelos HTR.

### Nombres de archivo
Los nombres de los archivos .txt siguen la nomenclatura:  
**{Archivo/Institución} _ {Fondo/Colección} _ {volumen/caja} _ {expediente/signatura} _ {página} _ {tipo de transcripción}** 
Por ejemplo: 
- AGI_CONTRATACION_5500_N2_R15_01_GT
  - se refiere a: Archivo General de Indias, Fondo Contratación, signatura 5500_N2_R15, página 01, transcripción Ground Truth
- AGN_HospJesus_c380_exp005_04_Redonda_m1t3_HTR
  - se refiere a: Archivo General de la Nación (México), Fondo Hospital de Jesús, caja 380, expediente 5, página 04, transcripción HTR hecha con el modelo Redonda m1t3

## Versiones del Corpus
### Corpus_oct2025
Esta fue la primera versión del Corpus. Incluye los documentos históricos utilizados para el entrenamiento de los modelos HTR para las letras Itálica cursiva, Procesal simple, Procesal encadenada y Redonda. Se incluyen las versiones GT y HTR obtenidas con sus respectivos modelos.
- Corpus_GT
  - 88 documentos/carpetas, 568 páginas/archivos .txt
- Corpus_HTR_Encadenada_m2t4
  - 6 documentos/carpetas, un solo archivo .txt por documento (no divididos en páginas individuales).
- Corpus_HTR_Italica_cursiva_m3t1
  - 47 documentos/carpetas, 335 páginas/archivos .txt
- Corpus_HTR_Procesal_m3t4
  - 24 documentos/carpetas, 146 páginas/archivos .txt
  - Nota: Esta no es la última versión del modelo para Procesal simple. La última versión es la m3t7 (ver más adelante).
- Corpus_HTR_Redonda_m1t3
  - 12 documentos/carpetas, 80 páginas/archivos .txt


### Corpus_feb2026
Esta versión del Corpus incluye los documentos históricos utilizados para el entrenamiento y la evaluación de los modelos HTR para las letras Itálica cursiva, Procesal simple, Procesal encadenada y Redonda. Se incluyen las versiones GT y HTR obtenidas con sus respectivos modelos.
- Corpus_GT
  - 157 documentos/carpetas, 2409 páginas/archivos .txt
- Corpus_HTR_Encadenada_m2t4
  - 15 documentos/carpetas, 716 páginas/archivos .txt          
- Corpus_HTR_Italica_cursiva_m3t1
  - 54 documentos/carpetas, 498 páginas/archivos .txt
- Corpus_HTR_Procesal_m3t7
  - 85 documentos/carpetas, 1102 páginas/archivos .txt
- Corpus_HTR_Redonda_m1t3
  - 19 documentos/carpetas, 97 páginas/archivos .txt

>[!CAUTION]
>Al analizar esta versión del Corpus con el proceso NAOMI, identificamos problemas de falta de alineamiento en 158 pares de archivos GT-HTR, donde uno de los dos txt tiene más renglones que el otro. 
>Estos afectaban a 76 itálica cursiva, 35 encadenada, 46 procesal y 1 redonda. Revisando cada par de archivos, los principales problemas identificados fueron:
>- Renglones extra en los GT.  
>El origen de este problema proviene del proceso de exportación de archivos .txt en Transkribus: cuando se exporta una transcripción del historial de versiones de un documento, junto con el texto de esa versión, se pega a continuación el texto de la última versión del historial. 
>Por lo tanto, si el GT no era la última versión, como es el caso en muchas de nuestras transcripciones en las que primero se hizo el GT y luego se corrieron uno o varios HTRs, entonces, al exportar ese GT, el txt incluye además el texto del último HTR del historial.
Se revisó y corrigió cada uno de los archivos para que sus renglones coincidieran.
>- AGI_CONTRATACION_1170A_N10.  
>Este documento tenía casi 100 pares GT-HTR con errores de concordancia en los renglones de texto, afectando a encadenada, cursiva y procesal. El problema surgió porque los HTR se corrieron sobre versiones casi terminadas de las transcripciones humanas que todavía no estaban completamente corregidas, por lo que muchas veces no correspondían con el GT final.
Para corregir los archivos afectados, se volvieron a correr los HTRs de este documento, ya en la versión final del GT.
>- Renglones presentes en GT pero ausentes en HTR.  
>Esto fue un caso extraño, exclusivo de los primeros documentos con los que trabajamos para el modelo de Itálica cursiva y casi siempre en documentos de BnF.
Algunos renglones tenían línea base, pero no polígono. Esto hizo que el renglón sí tuviera texto en el GT (pues se le agregó manualmente), pero no en el HTR pues no había polígono/imagen de entrada para el modelo.
Para corregirlo, se agregaron nuevas líneas base y polígonos a los renglones afectados. 
>- UIA_Ms149_Exp2.  
>En varias páginas de este documento, el GT no estaba completamente limpio. Se habían dejado renglones vacíos en los márgenes, pero no se habían eliminado, por lo que sí se incluyeron en los HTR, donde aparecían con texto.
Para corregirlo, se agregaron los textos a los GT donde hacía falta.


### Corpus_may2026
>[!NOTE]
>Esta es la versión más reciente y limpia del Corpus.  
En esta versión se han corregido todos los errores de alineamiento identificados en el Corpus_feb2026.  
Se incluyen los documentos históricos utilizados para el entrenamiento y la evaluación de las últimas versiones de los modelos HTR para las letras Itálica cursiva, Procesal simple, Procesal encadenada y Redonda. También se incluyen los utilizados en el modelo HTR para letra Pseudorredonda. 
- Corpus_GT
  - 173 documentos/carpetas, 2552 páginas/archivos .txt
- Corpus_HTR_Encadenada_m2t4
  - 15 documentos/carpetas, 716 páginas/archivos .txt
- Corpus_HTR_Italica_cursiva_m3t1
  - 54 documentos/carpetas, 498 páginas/archivos .txt
- Corpus_HTR_Procesal_m3t7
  - 85 documentos/carpetas, 1102 páginas/archivos .txt
- Corpus_HTR_Pseudorredonda_m1t1
  - 16 documentos/carpetas, 143 páginas/archivos .txt
- Corpus_HTR_Redonda_m1t3
  - 20 documentos/carpetas, 98 páginas/archivos .txt
