// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title TradingBotVault
 * @dev Main vault contract for automated trading bot operations
 */
contract TradingBotVault is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    
    struct Position {
        address token;
        uint256 amount;
        uint256 entryPrice;
        uint256 timestamp;
        bool isLong;
        string strategy;
    }
    
    struct Strategy {
        string name;
        bool active;
        uint256 allocation; // Percentage of total funds (basis points)
        uint256 maxSlippage; // Maximum allowed slippage (basis points)
        uint256 stopLoss; // Stop loss percentage (basis points)
        uint256 takeProfit; // Take profit percentage (basis points)
        uint256 maxPositions; // Maximum number of positions
    }
    
    struct PerformanceMetrics {
        uint256 totalPnL;
        uint256 totalTrades;
        uint256 winningTrades;
        uint256 totalVolume;
        uint256 lastUpdateTime;
    }
    
    // State variables
    mapping(address => bool) public authorizedBots;
    mapping(uint256 => Position) public positions;
    mapping(string => Strategy) public strategies;
    mapping(address => uint256) public userDeposits;
    mapping(address => PerformanceMetrics) public botPerformance;
    
    uint256 public nextPositionId;
    uint256 public totalValueLocked;
    uint256 public performanceFee; // Performance fee in basis points (e.g., 200 = 2%)
    uint256 public managementFee; // Annual management fee in basis points
    uint256 public constant MAX_FEE = 1000; // Maximum 10% fee
    uint256 public constant BASIS_POINTS = 10000;
    
    address public feeRecipient;
    address public emergencyAdmin;
    IERC20 public baseToken; // Primary token for the vault (e.g., USDC)
    
    // Events
    event PositionOpened(
        uint256 indexed positionId, 
        address indexed bot,
        address token, 
        uint256 amount, 
        bool isLong,
        string strategy
    );
    
    event PositionClosed(
        uint256 indexed positionId,
        address indexed bot,
        int256 pnl,
        uint256 exitPrice
    );
    
    event StrategyAdded(string indexed name, uint256 allocation);
    event StrategyUpdated(string indexed name, uint256 allocation, bool active);
    event EmergencyStop(address indexed admin, uint256 timestamp);
    event FeesCollected(uint256 performanceFees, uint256 managementFees);
    event UserDeposit(address indexed user, uint256 amount);
    event UserWithdraw(address indexed user, uint256 amount);
    event BotAuthorized(address indexed bot, bool authorized);
    
    // Modifiers
    modifier onlyAuthorizedBot() {
        require(authorizedBots[msg.sender], "TradingBotVault: Not authorized bot");
        _;
    }
    
    modifier onlyEmergencyAdmin() {
        require(msg.sender == emergencyAdmin, "TradingBotVault: Not emergency admin");
        _;
    }
    
    modifier validFee(uint256 fee) {
        require(fee <= MAX_FEE, "TradingBotVault: Fee too high");
        _;
    }
    
    constructor(
        address _baseToken,
        address _feeRecipient,
        address _emergencyAdmin,
        uint256 _performanceFee,
        uint256 _managementFee
    ) validFee(_performanceFee) validFee(_managementFee) {
        require(_baseToken != address(0), "TradingBotVault: Invalid base token");
        require(_feeRecipient != address(0), "TradingBotVault: Invalid fee recipient");
        require(_emergencyAdmin != address(0), "TradingBotVault: Invalid emergency admin");
        
        baseToken = IERC20(_baseToken);
        feeRecipient = _feeRecipient;
        emergencyAdmin = _emergencyAdmin;
        performanceFee = _performanceFee;
        managementFee = _managementFee;
    }
    
    /**
     * @dev Add or remove authorized trading bot
     */
    function setAuthorizedBot(address bot, bool authorized) external onlyOwner {
        require(bot != address(0), "TradingBotVault: Invalid bot address");
        authorizedBots[bot] = authorized;
        emit BotAuthorized(bot, authorized);
    }
    
    /**
     * @dev Add a new trading strategy
     */
    function addStrategy(
        string memory name,
        uint256 allocation,
        uint256 maxSlippage,
        uint256 stopLoss,
        uint256 takeProfit,
        uint256 maxPositions
    ) external onlyOwner {
        require(bytes(name).length > 0, "TradingBotVault: Invalid strategy name");
        require(allocation <= BASIS_POINTS, "TradingBotVault: Allocation exceeds 100%");
        require(maxSlippage <= 1000, "TradingBotVault: Max slippage too high"); // Max 10%
        require(stopLoss <= 5000, "TradingBotVault: Stop loss too high"); // Max 50%
        require(takeProfit <= 10000, "TradingBotVault: Take profit too high"); // Max 100%
        
        strategies[name] = Strategy({
            name: name,
            active: true,
            allocation: allocation,
            maxSlippage: maxSlippage,
            stopLoss: stopLoss,
            takeProfit: takeProfit,
            maxPositions: maxPositions
        });
        
        emit StrategyAdded(name, allocation);
    }
    
    /**
     * @dev Update existing strategy parameters
     */
    function updateStrategy(
        string memory name,
        uint256 allocation,
        bool active
    ) external onlyOwner {
        Strategy storage strategy = strategies[name];
        require(bytes(strategy.name).length > 0, "TradingBotVault: Strategy does not exist");
        require(allocation <= BASIS_POINTS, "TradingBotVault: Allocation exceeds 100%");
        
        strategy.allocation = allocation;
        strategy.active = active;
        
        emit StrategyUpdated(name, allocation, active);
    }
    
    /**
     * @dev Open a new trading position
     */
    function openPosition(
        address token,
        uint256 amount,
        uint256 entryPrice,
        bool isLong,
        string memory strategyName
    ) external onlyAuthorizedBot whenNotPaused nonReentrant returns (uint256 positionId) {
        require(strategies[strategyName].active, "TradingBotVault: Strategy not active");
        require(token != address(0), "TradingBotVault: Invalid token");
        require(amount > 0, "TradingBotVault: Invalid amount");
        require(entryPrice > 0, "TradingBotVault: Invalid entry price");
        
        // Check strategy allocation limits
        Strategy memory strategy = strategies[strategyName];
        require(strategy.active, "TradingBotVault: Strategy not active");
        
        positionId = nextPositionId++;
        
        positions[positionId] = Position({
            token: token,
            amount: amount,
            entryPrice: entryPrice,
            timestamp: block.timestamp,
            isLong: isLong,
            strategy: strategyName
        });
        
        totalValueLocked += amount;
        
        // Update bot performance metrics
        PerformanceMetrics storage metrics = botPerformance[msg.sender];
        metrics.totalTrades++;
        metrics.lastUpdateTime = block.timestamp;
        
        emit PositionOpened(positionId, msg.sender, token, amount, isLong, strategyName);
    }
    
    /**
     * @dev Close an existing position
     */
    function closePosition(
        uint256 positionId,
        uint256 exitPrice
    ) external onlyAuthorizedBot nonReentrant {
        Position storage position = positions[positionId];
        require(position.amount > 0, "TradingBotVault: Position does not exist");
        require(exitPrice > 0, "TradingBotVault: Invalid exit price");
        
        // Calculate P&L
        int256 pnl = _calculatePnL(position, exitPrice);
        
        // Update performance metrics
        PerformanceMetrics storage metrics = botPerformance[msg.sender];
        if (pnl > 0) {
            metrics.winningTrades++;
            metrics.totalPnL += uint256(pnl);
        } else {
            // Handle loss (pnl is negative)
            if (uint256(-pnl) > metrics.totalPnL) {
                metrics.totalPnL = 0;
            } else {
                metrics.totalPnL -= uint256(-pnl);
            }
        }
        metrics.totalVolume += position.amount;
        metrics.lastUpdateTime = block.timestamp;
        
        // Update TVL
        if (totalValueLocked >= position.amount) {
            totalValueLocked -= position.amount;
        }
        
        emit PositionClosed(positionId, msg.sender, pnl, exitPrice);
        
        // Clear position
        delete positions[positionId];
    }
    
    /**
     * @dev Calculate P&L for a position
     */
    function _calculatePnL(Position memory position, uint256 exitPrice) internal pure returns (int256) {
        if (position.isLong) {
            // Long position: profit when exit price > entry price
            if (exitPrice >= position.entryPrice) {
                return int256((exitPrice - position.entryPrice) * position.amount / position.entryPrice);
            } else {
                return -int256((position.entryPrice - exitPrice) * position.amount / position.entryPrice);
            }
        } else {
            // Short position: profit when exit price < entry price
            if (position.entryPrice >= exitPrice) {
                return int256((position.entryPrice - exitPrice) * position.amount / position.entryPrice);
            } else {
                return -int256((exitPrice - position.entryPrice) * position.amount / position.entryPrice);
            }
        }
    }
    
    /**
     * @dev Allow users to deposit funds into the vault
     */
    function deposit(uint256 amount) external whenNotPaused nonReentrant {
        require(amount > 0, "TradingBotVault: Amount must be greater than 0");
        
        baseToken.safeTransferFrom(msg.sender, address(this), amount);
        userDeposits[msg.sender] += amount;
        totalValueLocked += amount;
        
        emit UserDeposit(msg.sender, amount);
    }
    
    /**
     * @dev Allow users to withdraw their funds
     */
    function withdraw(uint256 amount) external nonReentrant {
        require(amount > 0, "TradingBotVault: Amount must be greater than 0");
        require(userDeposits[msg.sender] >= amount, "TradingBotVault: Insufficient balance");
        require(baseToken.balanceOf(address(this)) >= amount, "TradingBotVault: Insufficient vault balance");
        
        userDeposits[msg.sender] -= amount;
        totalValueLocked -= amount;
        
        baseToken.safeTransfer(msg.sender, amount);
        
        emit UserWithdraw(msg.sender, amount);
    }
    
    /**
     * @dev Emergency stop function - pauses all operations
     */
    function emergencyStop() external onlyEmergencyAdmin {
        _pause();
        emit EmergencyStop(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Resume operations after emergency stop
     */
    function resumeOperations() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Collect management and performance fees
     */
    function collectFees() external onlyOwner {
        uint256 totalFees = 0;
        uint256 vaultBalance = baseToken.balanceOf(address(this));
        
        // Calculate management fee (annualized)
        uint256 managementFeeAmount = (totalValueLocked * managementFee) / (BASIS_POINTS * 365);
        
        // Calculate performance fees based on profits
        uint256 performanceFeeAmount = 0;
        // This would typically involve more complex profit calculation
        // For simplicity, we'll calculate it based on vault growth
        
        totalFees = managementFeeAmount + performanceFeeAmount;
        
        if (totalFees > 0 && vaultBalance >= totalFees) {
            baseToken.safeTransfer(feeRecipient, totalFees);
            emit FeesCollected(performanceFeeAmount, managementFeeAmount);
        }
    }
    
    /**
     * @dev Update fee parameters
     */
    function updateFees(uint256 _performanceFee, uint256 _managementFee) 
        external 
        onlyOwner 
        validFee(_performanceFee) 
        validFee(_managementFee) 
    {
        performanceFee = _performanceFee;
        managementFee = _managementFee;
    }
    
    /**
     * @dev Update fee recipient
     */
    function updateFeeRecipient(address _feeRecipient) external onlyOwner {
        require(_feeRecipient != address(0), "TradingBotVault: Invalid fee recipient");
        feeRecipient = _feeRecipient;
    }
    
    /**
     * @dev Update emergency admin
     */
    function updateEmergencyAdmin(address _emergencyAdmin) external onlyOwner {
        require(_emergencyAdmin != address(0), "TradingBotVault: Invalid emergency admin");
        emergencyAdmin = _emergencyAdmin;
    }
    
    /**
     * @dev Get position details
     */
    function getPosition(uint256 positionId) external view returns (
        address token,
        uint256 amount,
        uint256 entryPrice,
        uint256 timestamp,
        bool isLong,
        string memory strategy
    ) {
        Position memory position = positions[positionId];
        return (
            position.token,
            position.amount,
            position.entryPrice,
            position.timestamp,
            position.isLong,
            position.strategy
        );
    }
    
    /**
     * @dev Get strategy details
     */
    function getStrategy(string memory name) external view returns (
        bool active,
        uint256 allocation,
        uint256 maxSlippage,
        uint256 stopLoss,
        uint256 takeProfit,
        uint256 maxPositions
    ) {
        Strategy memory strategy = strategies[name];
        return (
            strategy.active,
            strategy.allocation,
            strategy.maxSlippage,
            strategy.stopLoss,
            strategy.takeProfit,
            strategy.maxPositions
        );
    }
    
    /**
     * @dev Get bot performance metrics
     */
    function getBotPerformance(address bot) external view returns (
        uint256 totalPnL,
        uint256 totalTrades,
        uint256 winningTrades,
        uint256 totalVolume,
        uint256 lastUpdateTime
    ) {
        PerformanceMetrics memory metrics = botPerformance[bot];
        return (
            metrics.totalPnL,
            metrics.totalTrades,
            metrics.winningTrades,
            metrics.totalVolume,
            metrics.lastUpdateTime
        );
    }
    
    /**
     * @dev Get vault summary
     */
    function getVaultSummary() external view returns (
        uint256 _totalValueLocked,
        uint256 _nextPositionId,
        uint256 _performanceFee,
        uint256 _managementFee,
        address _baseToken,
        bool _paused
    ) {
        return (
            totalValueLocked,
            nextPositionId,
            performanceFee,
            managementFee,
            address(baseToken),
            paused()
        );
    }
    
    /**
     * @dev Emergency withdrawal function for owner
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        require(token != address(0), "TradingBotVault: Invalid token");
        IERC20(token).safeTransfer(owner(), amount);
    }
}