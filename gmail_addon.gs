/**
 * Gmail Add-on для Email Assistant
 * 
 * Установка:
 * 1. Открыть https://script.google.com
 * 2. Новый проект
 * 3. Скопировать этот код
 * 4. Создать appsscript.json (см. ниже)
 * 5. Deploy → Test deployment
 */

// === КОНФИГУРАЦИЯ ===
// ВАЖНО: Замени на адрес твоего API
const API_URL = 'http://localhost:5000'; // Для локального тестирования
// const API_URL = 'https://your-api.com'; // Для продакшена (Heroku, Railway, etc.)

// === ГЛАВНАЯ ФУНКЦИЯ ===
/**
 * Создает карточку для Gmail Add-on
 */
function buildAddOn(e) {
  var accessToken = e.messageMetadata.accessToken;
  var messageId = e.messageMetadata.messageId;
  GmailApp.setCurrentMessageAccessToken(accessToken);
  
  var message = GmailApp.getMessageById(messageId);
  var subject = message.getSubject();
  var body = message.getPlainBody();
  
  // Классификация письма
  var classification = classifyEmail(subject + " " + body);
  
  // Создаем UI
  var card = CardService.newCardBuilder();
  
  // Заголовок
  card.setHeader(CardService.newCardHeader()
    .setTitle('Email Assistant')
    .setSubtitle('AI-powered email processing')
    .setImageUrl('https://www.gstatic.com/images/branding/product/1x/gmail_48dp.png'));
  
  // Секция классификации
  var classificationSection = CardService.newCardSection()
    .setHeader('<b>Email Classification</b>');
  
  classificationSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Category')
      .setContent(classification.category.toUpperCase())
      .setIcon(getCategoryIcon(classification.category))
  );
  
  classificationSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Confidence')
      .setContent(classification.confidence + '%')
  );
  
  // Кнопка применить метку
  classificationSection.addWidget(
    CardService.newTextButton()
      .setText('Apply Label')
      .setOnClickAction(CardService.newAction()
        .setFunctionName('applyLabel')
        .setParameters({
          'messageId': messageId,
          'category': classification.category
        }))
  );
  
  card.addSection(classificationSection);
  
  // Секция для создания ответа
  var composeSection = CardService.newCardSection()
    .setHeader('<b>Compose Reply</b>');
  
  var draftInput = CardService.newTextInput()
    .setFieldName('draft')
    .setTitle('Your draft (short version)')
    .setHint('e.g., "OK, I will come tomorrow"')
    .setMultiline(true);
  
  composeSection.addWidget(draftInput);
  
  composeSection.addWidget(
    CardService.newTextButton()
      .setText('Generate Reply')
      .setOnClickAction(CardService.newAction()
        .setFunctionName('generateReply')
        .setParameters({
          'category': classification.category
        }))
  );
  
  card.addSection(composeSection);
  
  return card.build();
}

/**
 * Классификация email через API
 */
function classifyEmail(text) {
  try {
    var response = UrlFetchApp.fetch(API_URL + '/classify', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({ text: text }),
      muteHttpExceptions: true
    });
    
    var result = JSON.parse(response.getContentText());
    return result;
  } catch (error) {
    Logger.log('Error classifying email: ' + error);
    return {
      category: 'unknown',
      confidence: 0,
      all_probabilities: {}
    };
  }
}

/**
 * Применить метку к письму
 */
function applyLabel(e) {
  var messageId = e.parameters.messageId;
  var category = e.parameters.category;
  
  try {
    var message = GmailApp.getMessageById(messageId);
    var labelName = 'AI/' + category.toUpperCase();
    
    // Создать метку если не существует
    var label = GmailApp.getUserLabelByName(labelName);
    if (!label) {
      label = GmailApp.createLabel(labelName);
    }
    
    // Применить метку
    message.getThread().addLabel(label);
    
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification()
        .setText('Label "' + labelName + '" applied successfully!'))
      .build();
  } catch (error) {
    Logger.log('Error applying label: ' + error);
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification()
        .setText('Error: ' + error))
      .build();
  }
}

/**
 * Генерация ответа
 */
function generateReply(e) {
  var draft = e.formInput.draft;
  var category = e.parameters.category;
  
  if (!draft) {
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification()
        .setText('Please enter a draft text'))
      .build();
  }
  
  try {
    var response = UrlFetchApp.fetch(API_URL + '/compose', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({
        draft: draft,
        sender_name: Session.getActiveUser().getEmail().split('@')[0],
        recipient_name: null
      }),
      muteHttpExceptions: true
    });
    
    var result = JSON.parse(response.getContentText());
    
    // Создаем новую карточку с результатом
    var card = CardService.newCardBuilder();
    
    card.setHeader(CardService.newCardHeader()
      .setTitle('Generated Reply')
      .setSubtitle('Category: ' + result.category));
    
    var resultSection = CardService.newCardSection();
    
    resultSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>Your draft:</b><br>' + result.draft)
    );
    
    resultSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>Generated email:</b><br><pre>' + result.full_email + '</pre>')
    );
    
    // Кнопка копировать
    resultSection.addWidget(
      CardService.newTextButton()
        .setText('Copy to Clipboard')
        .setOpenLink(CardService.newOpenLink()
          .setUrl('https://www.google.com')
          .setOpenAs(CardService.OpenAs.OVERLAY))
    );
    
    card.addSection(resultSection);
    
    return CardService.newActionResponseBuilder()
      .setNavigation(CardService.newNavigation().pushCard(card.build()))
      .build();
      
  } catch (error) {
    Logger.log('Error generating reply: ' + error);
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification()
        .setText('Error: ' + error))
      .build();
  }
}

/**
 * Получить иконку для категории
 */
function getCategoryIcon(category) {
  var icons = {
    'work': CardService.Icon.BOOKMARK,
    'personal': CardService.Icon.PERSON,
    'spam': CardService.Icon.STAR,
    'promo': CardService.Icon.OFFER
  };
  return icons[category] || CardService.Icon.EMAIL;
}

/**
 * Функция для homepage (опционально)
 */
function buildHomepage(e) {
  var card = CardService.newCardBuilder();
  
  card.setHeader(CardService.newCardHeader()
    .setTitle('Email Assistant')
    .setSubtitle('AI-powered email classification and composition'));
  
  var section = CardService.newCardSection();
  
  section.addWidget(
    CardService.newTextParagraph()
      .setText('Open any email to use Email Assistant features:<br><br>' +
               '• Automatic classification<br>' +
               '• Smart reply generation<br>' +
               '• Style adaptation')
  );
  
  card.addSection(section);
  
  return card.build();
}
