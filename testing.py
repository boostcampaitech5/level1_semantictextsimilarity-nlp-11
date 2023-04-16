import nlpaug.augmenter.word as naw
import nlpaug.model.language_models as nml

korean_text = '안녕하세요, 반갑습니다'

translator = nml.KoBART()
aug = naw.ContextualWordEmbsAug(model_path=translator, action="substitute")
translated_text = aug.augment(korean_text)

print(translated_text)