# =================================================================================
# DICIONÁRIO HIERÁRQUICO UNIFICADO (vFinal)
# Inclui os Termos Gerais e os termos específicos do Item 8.4 do Formulário de Referência.
# Este dicionário deve ser usado como a única fonte para o enriquecimento de metadados.
# =================================================================================
DICIONARIO_UNIFICADO_HIERARQUICO = {
    
    # ==========================================================================
    # Seção: Termos específicos do Formulário de Referência (Item 8.4)
    # ==========================================================================
    "FormularioReferencia_Item_8_4": {
        "a_TermosGerais": ["termos e condições gerais", "objetivos do plano", "elegíveis", "principais regras"],
        "b_Aprovacao": ["data de aprovação", "órgão responsável", "conselho de administração", "assembleia geral"],
        "c_MaximoAcoes": ["número máximo de ações abrangidas", "diluição máxima"],
        "d_MaximoOpcoes": ["número máximo de opções a serem outorgadas", "limite de opções"],
        "e_CondicoesAquisicao": ["condições de aquisição de ações", "metas de desempenho", "tempo de serviço"],
        "f_CriteriosPreco": ["critérios para fixação do preço de aquisição", "preço de exercício", "preço fixo previamente estabelecido"],
        "g_CriteriosPrazo": ["critérios para fixação do prazo de aquisição", "prazo de exercício"],
        "h_FormaLiquidacao": ["forma de liquidação", "pagamento em dinheiro", "entrega física das ações"],
        "i_RestricoesTransferencia": ["restrições à transferência", "períodos de bloqueio", "lockup"],
        "j_SuspensaoExtincao": ["suspensão, alteração ou extinção do plano", "mudanças nas políticas", "desempenho financeiro"],
        "k_EfeitosSaida": ["efeitos da saída do administrador", "regras de desligamento", "aposentadoria", "demissão"],
    },

    # ==========================================================================
    # Seção: Define os tipos de planos de incentivo existentes.
    # ==========================================================================
    "TiposDePlano": {
        "AcoesRestritas": ["Restricted Shares", "Plano de Ações Restritas", "RSU", "ações restritas"],
        "OpcoesDeCompra": ["Stock Options", "ESOP", "Plano de Opção de Compra", "opções de compra", "SOP"],
        "AcoesFantasmas": ["Phantom Shares", "Ações Virtuais", "virtuais"],
        "OpcoesFantasmas_SAR": ["Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
        "PlanoCompraAcoes_ESPP": ["Plano de Compra de Ações", "Employee Stock Purchase Plan", "ESPP"],
        "BonusRetencaoDiferido": ["Staying Bonus", "Retention Bonus", "Bônus de Permanência", "Bônus de Retenção", "Deferred Bonus"],
        "Matching_Coinvestimento": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching"],
    },

    # ==========================================================================
    # Seção: Descreve as mecânicas e o ciclo de vida de um plano.
    # ==========================================================================
    "MecanicasCicloDeVida": {
        "Outorga": ["Outorga", "Concessão", "Grant", "Grant Date"],
        "Vesting": ["Vesting", "Período de Carência", "Aquisição de Direitos", "cronograma de vesting", "Vesting Gradual"],
        "VestingCliff": ["Cliff", "Cliff Period", "Período de Cliff", "Carência Inicial"],
        "VestingAcelerado": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação do vesting"],
        "VestingTranche": ["Tranche", "Lote", "Parcela do Vesting", 'Parcela', "Aniversário"],
        "PrecoExercicio": ["Preço de Exercício", "Strike", "Strike Price"],
        "PrecoDesconto": ["Desconto de", "preço com desconto", "desconto sobre o preço"],
        "CicloExercicio": ["Exercício", "Período de Exercício", "pagamento", "liquidação", "vencimento", "expiração"],
        "Lockup": ["Lockup", "Período de Lockup", "Restrição de Venda"],
    },

    # ==========================================================================
    # Seção: Termos relacionados à governança e gestão de riscos.
    # ==========================================================================
    "GovernancaRisco": {
        "DocumentosPlano": ["Regulamento", "Regulamento do Plano", "Contrato de Adesão", "Termo de Outorga"],
        "OrgaoDeliberativo": ["Comitê de Remuneração", "Comitê de Pessoas", "Deliberação do Conselho"],
        "MalusClawback": ["Malus", "Clawback", "Redução", "Devolução", "Cláusula de Recuperação", "Forfeiture"],
        "Diluicao": ["Diluição", "Dilution", "Capital Social"],
    },

    # ==========================================================================
    # Seção: Define quem participa e sob quais condições.
    # ==========================================================================
    "ParticipantesCondicoes": {
        "Elegibilidade": ["Participantes", "Beneficiários", "Elegíveis", "Empregados", "Administradores", "Colaboradores"],
        "CondicaoSaida": ["Desligamento", "Saída", "Término do Contrato", "Rescisão", "Demissão", "Good Leaver", "Bad Leaver"],
        "CasosEspeciais": ["Aposentadoria", "Morte", "Invalidez", "Afastamento"],
    },

    # ==========================================================================
    # Seção: Métricas de performance usadas para atrelar condições ao plano.
    # ==========================================================================
    "IndicadoresPerformance": {
        "MetasGerais": ["Performance Shares", "Performance Units", "PSU", "Plano de Desempenho", "Metas de Performance"],
        "Financeiro": ["ROIC", "EBITDA", "LAIR", "Lucro", "CAGR", "Receita Líquida"],
        "Mercado": ["CDI", "IPCA", "Selic"],
        "TSR_Absoluto": ["TSR", "Total Shareholder Return", "Retorno Total ao Acionista"],
        "TSR_Relativo": ["TSR Relativo", "Relative TSR", "TSR versus", "TSR comparado a"],
        "GrupoDeComparacao": ["Peer Group", "Empresas Comparáveis", "Companhias Comparáveis"],
        "ESG": ["Metas ESG", "ESG"],
    },
    
    # ==========================================================================
    # Seção: Eventos corporativos e financeiros que afetam os planos.
    # ==========================================================================
    "EventosFinanceiros": {
        "EventosCorporativos": ["IPO", "grupamento", "desdobramento", "cisão", "fusão", "incorporação", "bonificação"],
        "MudancaDeControle": ["Mudança de Controle", "Change of Control", "Evento de Liquidez"],
        "DividendosProventos": ["Dividendos", "JCP", "Juros sobre capital próprio", "dividend equivalent", "proventos"],
    },

    # ==========================================================================
    # Seção: Termos relacionados a impostos, encargos e normas contábeis.
    # ==========================================================================
    "AspectosFiscaisContabeis": {
        "TributacaoEncargos": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"],
        "NormasContabeis": ["IFRS 2", "CPC 10", "Valor Justo", "Fair Value", "Black-Scholes", "Despesa Contábil", "Volatilidade"],
    }
}
